import os
import torch
from torch import nn
import argparse
from tqdm import tqdm
from model.multipathpp import MultiPathPP
from model.data import get_dataloader
from utils.utils import get_yaml_config, dict_to_cuda
from model.normlization import normalize


# It's a inverse operation of _transfrom_to_agent_coordinate_system in prerender/vectorizer.py
def convert_to_raw_xy(coordinates, yaw, shift):
    yaw = -yaw
    c, s = torch.cos(yaw), torch.sin(yaw)
    R = torch.cat([c, -s, s, c], axis=-1)
    R = R.reshape(*R.shape[: -1], 2, 2)
    # [b, n, t, 2] -> [b, n, t, 1, 2]
    coordinates = coordinates.unsqueeze(-2)
    coordinates = torch.matmul(coordinates, R).squeeze(-2) + shift
    return coordinates


def rollout(args):
    config = get_yaml_config(args.config)
    dataloader = get_dataloader(args.test_data_path, config["test"]["data_config"])
    model = MultiPathPP(config["model"])
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    model.eval()
    last_file_name = ""
    outputs = {}
    with torch.no_grad():
        for data in tqdm(dataloader):
            if config["test"]["data_config"]["dataset_config"]["normlization"]:
                data = normalize(data)
            dict_to_cuda(data)
            # coordinates is [b, n, m, t, 2]
            # yaws is [b, n, m, t, 1]
            # probas is [b, n, m]
            probas, coordinates, yaws = model(data)
            b, n, m, t, _  = coordinates.shape
            probas = nn.functional.softmax(probas.reshape(-1, probas.shape[-1])).reshape(b, n, m)
            coordinates = convert_to_raw_xy(
                coordinates + data["history/xy"][:, :, -1:, :].unsqueeze(2), 
                data["yaw"].cuda().reshape(b, 1, 1, 1, 1), 
                data["shift"].cuda().squeeze().reshape(b, 1, 1, 1, 2),
            )
            """
            # expand best traj to all modes
            if args.use_best_traj:
                _, select_id = torch.max(probas, axis=-1)
                select_id = select_id.unsqueeze([-1, -2, -3])
                coordinates = torch.gather(
                    coordinates, 
                    index=select_id.repeat(1, 1, 1, t, 2), 
                    dim=-2)
                coordinates = coordinates.repeat(1, 1, m, 1, 1)
                yaws = torch.gather(
                    yaws, 
                    index=select_id.repeat(1, 1, 1, t, 1), 
                    dim=-2)
                yaws = yaws.repeat(1, 1, m, 1, 1)
            """
            # get yaw
            if args.predict_yaw == "predict":
                yaws = yaws + data["history/yaw"][:, :, -1:, :].unsqueeze(2).repeat(1, 1, m, t, 1) + data["yaw"].reshape(b, 1, 1, 1, 1)
            elif args.predict_yaw == "current":
                yaws = data["history/yaw"][:, :, -1:, :].unsqueeze(2).repeat(1, 1, m, t, 1) + data["yaw"].reshape(b, 1, 1, 1, 1)
            elif args.predict_yaw == "interpolation":
                history_xy = convert_to_raw_xy(
                    data["history/xy"], 
                    data["yaw"].cuda().reshape(b, 1, 1, 1), 
                    data["shift"].cuda().reshape(b, 1, 1, 2)
                ).unsqueeze(2).repeat(1, 1, m, 1, 1)
                xy = torch.cat([history_xy[:, :, :, -1:, :], coordinates[:, :, :, : -1, :]], axis=-2)
                d_xy = coordinates - xy
                yaws = torch.atan2(d_xy[:, :, :, :, 1], d_xy[:, :, :, :, 0]).unsqueeze(-1) + torch.acos(torch.zeros(1)).item() * 2
                yaws_smooth = (yaws[:, :, :, : -1, :] + yaws[:, :, :, 1: , :]) / 2
                yaws = torch.cat([yaws_smooth, yaws[:, :, :, -1: , :]], axis=-2)
            else:
                raise ValueError(f"{args.predict_yaw} is not supported")
            coordinates_z = data["agent_z"].cuda().reshape(b, n, 1, 1, 1).repeat(1, 1, m, t, 1)
            coordinates = torch.cat([coordinates, coordinates_z], axis=-1)
            
            for i in range(len(data["scenario_id"])):
                senario_id = data["scenario_id"][i]
                file_name = data["file_name"][i]
                senario_output = {
                    "agent_id": data["agent_id"][i],
                    "xyz": coordinates[i].permute(1, 0, 2, 3).cpu(),
                    "yaws": yaws[i].permute(1, 0, 2, 3).cpu(),
                    "probs": probas[i].permute(1, 0).cpu(),
                }
                assert senario_output["agent_id"].shape == (n, )
                assert senario_output["xyz"].shape == (m, n, t, 3)
                assert senario_output["yaws"].shape == (m, n, t, 1)
                assert senario_output["probs"].shape == (m, n)
                
                if file_name == last_file_name:
                    outputs[senario_id] = senario_output
                else:
                    if last_file_name != "":
                        torch.save(outputs, os.path.join(args.save_path, f"{last_file_name}.pth"))
                    last_file_name = file_name
                    outputs = {senario_id: senario_output}
        torch.save(outputs, os.path.join(args.save_path, f"{last_file_name}.pth"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--predict_yaw", type=str, required=False, default="current")
    parser.add_argument("--config", type=str, required=False, default="configs/Multipathpp.yaml", help="Vectorizer Config")
    parser.add_argument("--use_best_traj", type=bool, required=False, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    rollout(args)
