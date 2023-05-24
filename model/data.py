import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def angle_to_range(yaw):
    yaw = (yaw - np.pi) % (2 * np.pi) - np.pi
    return yaw


class SegmentAndAgentSequenceDataset(Dataset):
    def __init__(self, data_path, config):
        self._data_path = data_path
        self._config = config
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]
        if "max_length" in config:
            self._files = self._files[:config["max_length"]]
        assert len(self._files) > 0
    
    def __len__(self):
        return len(self._files)
    
    def _generate_sin_cos(self, data):
        data["history/yaw_sin"] = np.sin(data["history/yaw"])
        data["history/yaw_cos"] = np.cos(data["history/yaw"])
        return data
    
    def _add_length_width(self, data):
        data["history/length"] = \
            data["length"].reshape(-1, 1, 1) * np.ones_like(data["history/yaw"])
        data["history/width"] = \
            data["width"].reshape(-1, 1, 1) * np.ones_like(data["history/yaw"])
        return data
    
    def _compute_agent_diff_features(self, data):
        diff_keys = ["history/xy", "history/yaw", "history/speed"]
        for key in diff_keys:
            if key.endswith("yaw"):
                # data[f"{key}_diff"] = angle_to_range(np.diff(data[key], axis=1))
                data[f"{key}_diff"] = np.diff(data[key], axis=1)
            else:
                data[f"{key}_diff"] = np.diff(data[key], axis=1)
        data["history/yaw_sin_diff"] = np.sin(data["history/yaw_diff"])
        data["history/yaw_cos_diff"] = np.cos(data["history/yaw_diff"])
        data["history/valid_diff"] = (data["history/valid"] * \
            np.concatenate([data["history/valid"][:, 1:, :],
            np.zeros((data["history/valid"].shape[0], 1, 1))], axis=1))[:, :-1, :]
        return data
    
    def _compute_agent_type_and_is_sdc_ohe(self, data):
        I = np.eye(5)
        agent_type_ohe = I[np.array(data["agent_type"])]
        is_sdc = np.zeros(agent_type_ohe.shape[0])
        is_sdc[data["ego_id"]] = 1
        ohe_data = np.concatenate([agent_type_ohe, is_sdc.reshape(-1, 1)], axis=-1)[:, None, :]
        ohe_data = np.repeat(ohe_data, data["history/xy"].shape[1], axis=1)
        return ohe_data
    
    def _mask_history(self, ndarray, fraction):
        assert fraction >= 0 and fraction < 1
        ndarray = ndarray * (np.random.uniform(size=ndarray.shape) > fraction)
        return ndarray
    
    def _compute_lstm_input_data(self, data):
        keys_to_stack = self._config["lstm_input_data"]
        keys_to_stack_diff = self._config["lstm_input_data_diff"]
        agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data)
        # lstm data
        data["history/lstm_data"] = np.concatenate(
            [data[f"history/{k}"] for k in keys_to_stack] + [agent_type_ohe], axis=-1)
        data["history/lstm_data"] *= data["history/valid"]
        # lstm data diff
        data["history/lstm_data_diff"] = np.concatenate(
            [data[f"history/{k}_diff"] for k in keys_to_stack_diff] + \
                [agent_type_ohe[:, 1:, :]], axis=-1)
        data["history/lstm_data_diff"] *= data["history/valid_diff"]
        return data

    def _compute_mcg_input_data(self, data):
        lstm_input_data = data["history/lstm_data"]
        I = np.eye(lstm_input_data.shape[1])[None, ...]
        timestamp_ohe = np.repeat(I, lstm_input_data.shape[0], axis=0)
        data["history/mcg_input_data"] = np.concatenate(
            [lstm_input_data, timestamp_ohe], axis=-1)
        return data
    
    def __getitem__(self, idx):
        np_data = dict(np.load(self._files[idx], allow_pickle=True))
        # for uncompressed data
        if "arr_0" in np_data:
            np_data = np_data["arr_0"].item()

        np_data["scenario_id"] = np_data["scenario_id"]
        np_data["filename"] = self._files[idx]
        # np_data["history/yaw"] = angle_to_range(np_data["history/yaw"])
        np_data["history/yaw"] = np_data["history/yaw"]
        np_data = self._generate_sin_cos(np_data)
        np_data = self._add_length_width(np_data)
        if self._config["mask_history"]:
            np_data["history/valid"] = self._mask_history(
                np_data["history/valid"], self._config["mask_history_fraction"])
        np_data = self._compute_agent_diff_features(np_data)
        np_data = self._compute_lstm_input_data(np_data)
        np_data = self._compute_mcg_input_data(np_data)
        return np_data

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}

        for data in batch:
            for k in batch_keys:
                if not isinstance(data[k], str) and len(data[k].shape) == 0:
                    result_dict[k].append(data[k].item())
                else:
                    result_dict[k].append(data[k])

        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            result_dict[k] = torch.Tensor(np.stack(v, axis=0))
        result_dict["batch_size"] = len(batch)
        return result_dict


def get_dataloader(data_path, config):
    dataset = SegmentAndAgentSequenceDataset(data_path, config["dataset_config"])
    dataloader = DataLoader(
        dataset, collate_fn=SegmentAndAgentSequenceDataset.collate_fn, **config["dataloader_config"])
    return dataloader