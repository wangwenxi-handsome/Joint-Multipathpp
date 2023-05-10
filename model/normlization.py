import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
import torch
import numpy as np
import argparse
from tqdm import tqdm
from model.data import get_dataloader
from utils.utils import get_yaml_config


normalizarion_means = {
    "history/lstm_data": np.array([3.6018e+00,1.4909e+00,-3.4022e-03,4.7697e+00,1.9604e+00,4.3876e+00,0,0,0,0,0,0,0],dtype=np.float32),
    "history/lstm_data_diff": np.array([0.2246,-0.0044,0.0003,-0.0003,0,0,0,0,0,0,0],dtype=np.float32),
    "history/mcg_input_data": np.array([3.6018e+00,1.4909e+00,-3.4022e-03,4.7697e+00,1.9604e+00,4.3876e+00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32),
    "road_network_embeddings": np.array([3.6018e+00,1.4909e+00,-3.4022e-03,1.9266e+02,1.3466e-02,5.4262e-02,1.0721e-01,-1.1991e-02,6.1582e+00,3.0463e+00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32),
}

normalizarion_stds = {
    "history/lstm_data": np.array([29.7143,20.0370,1.5417,6.0304,0.5791,1.5877,1,1,1,1,1,1,1],dtype=np.float32),
    "history/lstm_data_diff": np.array([0.6929,0.2720,0.0719,0.5725,1,1,1,1,1,1,1],dtype=np.float32),
    "history/mcg_input_data": np.array([29.7143,20.0370,1.5417,6.0304,0.5791,1.5877,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=np.float32),
    "road_network_embeddings": np.array([29.7143,20.0370,1.5417,1.3049e+03,6.9937e-01,7.1257e-01,8.1720e-01,5.6312e-01,1.2635e+00,3.0021e+00,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=np.float32),
}


def normalize(data, ):
    for k in normalizarion_means:
        data[k] = (data[k] - normalizarion_means[k]) / (normalizarion_stds[k] + 1e-6)
        data[k].clamp_(-15, 15)
    data[f"history/lstm_data_diff"] *= data[f"history/valid_diff"]
    data[f"history/lstm_data"] *= data[f"history/valid"]
    return data


def compute_data_normalization(args):
    batch_num = 0
    config = get_yaml_config(args.config)
    config["train"]["data_config"]["dataloader_config"]["num_workers"] = 8
    dataloader = get_dataloader(args.data_path, config["train"]["data_config"])

    # [xy, yaw, speed, width, length]
    agent_feature = []
    # [xy, yaw, speed]
    agent_diff_feature = []
    # [r_norm, r_unit_vector, segment_unit_vector, segment_end_minus_start_norm, segment_end_minus_r_norm]
    road_network_feature = []
    # xy
    future_diff_xy = []

    for data in tqdm(dataloader):
        batch_num += 1
        agent_feature.append(data["history/lstm_data"][:, : 9, :, :6])
        agent_diff_feature.append(data["history/lstm_data_diff"][:, : 9, :, :4])
        road_network_feature.append(data["road_network_embeddings"][:, : 9, : 64, 3: 10])
        future_diff_xy.append(data["future/xy"][:, :9, :, :] - data["history/xy"][:, :9, -1:, :])
        if batch_num >= args.batch_num:
            break
    
    print("future diff xy")
    future_diff_xy = torch.cat(future_diff_xy, axis=0).reshape(-1, 2)
    print(torch.mean(future_diff_xy, axis=0))
    print(torch.std(future_diff_xy, axis=0)) 

    print("agent_feature [xy, yaw, speed, width, length]")
    agent_feature = torch.cat(agent_feature, axis=0).reshape(-1, 6)
    print(torch.mean(agent_feature, axis=0))
    print(torch.std(agent_feature, axis=0))

    print("agent_diff_feature [xy, yaw, speed]")
    agent_diff_feature = torch.cat(agent_diff_feature, axis=0).reshape(-1, 4)
    print(torch.mean(agent_diff_feature, axis=0))
    print(torch.std(agent_diff_feature, axis=0))

    print("road_network_feature [r_norm, r_unit_vector, segment_unit_vector, segment_end_minus_start_norm, segment_end_minus_r_norm]")
    road_network_feature = torch.cat(road_network_feature, axis=0).reshape(-1, 7)
    print(torch.mean(road_network_feature, axis=0))
    print(torch.std(road_network_feature, axis=0))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data folder")
    parser.add_argument("--config", type=str, required=False, default="configs/Multipathpp.yaml", help="Vectorizer Config")
    parser.add_argument("--batch_num", type=int, required=False, default=80, help="max batch num")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    compute_data_normalization(args)