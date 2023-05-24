import torch
from torch import nn
from .modules import MCGBlock, HistoryEncoder, MLP, NormalMLP, MCGDecoder, MLPDecoder
from utils.utils import mask_by_valid

class MultiPathPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._agent_mcg_linear = NormalMLP(config["agent_mcg_linear"])
        self._agent_history_encoder = HistoryEncoder(config["agent_history_encoder"])
        self._agent_info_linear = MLP(config["agent_info_linear"])
        self._interaction_mcg_encoder = MCGBlock(config["interaction_mcg_encoder"])
        self._agent_intention_linear = NormalMLP(config["agent_intention_linear"])
        self._polyline_encoder = NormalMLP(config["polyline_encoder"])
        self._roadgraph_mcg_encoder = MCGBlock(config["roadgraph_mcg_encoder"])
        self._agent_linear = MLP(config["agent_linear"])
        # self._agent_mcg_encoder = MCGBlock(config["agent_mcg_encoder"])
        if config["decoder"] == "MCGDecoder":
            self._decoder = MCGDecoder(config["n_trajectories"], config["decoder_config"]["MCGDecoder"])
        elif config["decoder"] == "MLPDecoder":
            self._decoder = MLPDecoder(config["n_trajectories"], config["decoder_config"]["MLPDecoder"])
    
    def forward(self, data, num_steps=None):
        # Encoder
        # mcg_input_data_linear is [b, n, t, 128]
        mcg_input_data_linear = self._agent_mcg_linear(data["history/mcg_input_data"])
        assert torch.isfinite(mcg_input_data_linear).all()
        # agents_info_embeddings is [b, n, 256]
        agents_info_embeddings = self._agent_history_encoder(
            data["history/lstm_data"], data["history/lstm_data_diff"],
            mcg_input_data_linear)
        agents_info_embeddings = self._agent_info_linear(agents_info_embeddings)
        agents_info_embeddings = mask_by_valid(agents_info_embeddings, data["agent_valid"])
        assert torch.isfinite(agents_info_embeddings).all()
        # agent_interaction_embedding is [b, n, 256]
        # s is [b, n, n, 256], c is [b, n, 256](it will be expanded to [b, n, 1, 256] in MCG)
        agent_interaction_embedding = self._interaction_mcg_encoder(
            agents_info_embeddings.unsqueeze(1).repeat(1, agents_info_embeddings.shape[1], 1, 1), 
            agents_info_embeddings, return_s=False)
        agent_interaction_embedding = mask_by_valid(agent_interaction_embedding, data["agent_valid"])
        assert torch.isfinite(agent_interaction_embedding).all()
        # agent_intention_embedding is [b, n, 128]
        agent_intention_embedding = torch.cat(
            [agents_info_embeddings, agent_interaction_embedding], axis=-1)
        agent_intention_embedding = self._agent_intention_linear(agent_intention_embedding)
        agent_intention_embedding = mask_by_valid(agent_intention_embedding, data["agent_valid"])
        assert torch.isfinite(agent_intention_embedding).all()
        # segment_embeddings is [b, n, s, 128]
        segment_embeddings = self._polyline_encoder(data["road_network_embeddings"])
        segment_embeddings = mask_by_valid(segment_embeddings, data["road_segments_valid"])
        assert torch.isfinite(segment_embeddings).all()
        # roadgraph_mcg_embedding is [b, n, 128]
        # s is [b, n, s, 128], c is [b, n, 128](it will be expanded to [b, n, 1, 128] in MCG)
        roadgraph_mcg_embedding = self._roadgraph_mcg_encoder(
            segment_embeddings, agent_intention_embedding, return_s=False)
        roadgraph_mcg_embedding = mask_by_valid(roadgraph_mcg_embedding, data["agent_valid"])
        assert torch.isfinite(roadgraph_mcg_embedding).all()
        # agent_embedding is [b, n, 512]
        agent_embedding = torch.cat(
            [agents_info_embeddings, agent_intention_embedding, roadgraph_mcg_embedding], dim=-1)
        agent_embedding = self._agent_linear(agent_embedding)
        agent_embedding = mask_by_valid(agent_embedding, data["agent_valid"])
        """
        agent_embedding = self._agent_mcg_encoder(
            agent_embedding.unsqueeze(1).repeat(1, agent_embedding.shape[1], 1, 1), 
            agent_embedding, return_s=False)
        agent_embedding = mask_by_valid(agent_embedding, data["agent_valid"])
        """
        assert torch.isfinite(agent_embedding).all()

        # Decoder
        # probas is [b, n, m]
        # coordinates is [b, n, m, t, 2]
        # covariance_matrices is [b, n, m, t, 2, 2]
        # heading is [b, n, m, t, 1]
        probas, coordinates, yaws = self._decoder(agent_embedding)
        return probas, coordinates, yaws
