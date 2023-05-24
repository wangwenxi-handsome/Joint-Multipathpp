import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.multipathpp import MultiPathPP
from model.data import get_dataloader
from model.losses import get_model_loss
from utils.utils import set_random_seed, get_last_file, dict_to_cuda, get_yaml_config, mask_by_valid
from model.normlization import normalize


def train(args):
    # config
    set_random_seed(42)
    config = get_yaml_config(args.config)

    # dataloader
    dataloader = get_dataloader(args.train_data_path, config["train"]["data_config"])
    val_dataloader = get_dataloader(args.val_data_path, config["val"]["data_config"])

    # model init
    loss_func = get_model_loss(config["model"]["loss"])
    if(not os.path.exists(args.save_folder)):
        os.mkdir(args.save_folder)
    last_checkpoint = get_last_file(args.save_folder)
    model = MultiPathPP(config["model"])
    model.cuda()
    optimizer = Adam(model.parameters(), **config["train"]["optimizer"])
    if config["train"]["scheduler"]:
        scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
    num_steps = 0
    if last_checkpoint is not None:
        model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
        optimizer.load_state_dict(torch.load(last_checkpoint)["optimizer_state_dict"])
        num_steps = torch.load(last_checkpoint)["num_steps"]
        if config["train"]["scheduler"]:
            scheduler.load_state_dict(torch.load(last_checkpoint)["scheduler_state_dict"])
        print("LOADED ", last_checkpoint)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("N PARAMS=", params)

    # train and validation
    best_loss = float('inf')
    for epoch in tqdm(range(config["train"]["n_epochs"])):
        pbar = tqdm(dataloader)
        for data in pbar:
            # train
            model.train()
            optimizer.zero_grad()
            if config["train"]["data_config"]["dataset_config"]["normlization"]:
                data = normalize(data)
            dict_to_cuda(data)
            probas, coordinates, yaws = model(data, num_steps)

            # loss and optimizer
            gt_xy = data["future/xy"] - data["history/xy"][:, :, -1:, :]
            gt_yaw = data["future/yaw"] - data["history/yaw"][:, :, -1:, :]
            gt_valid = mask_by_valid(data["future/valid"], data["agent_valid"])
            distance_loss, yaw_loss, confidence_loss = loss_func(
                gt_xy, gt_valid, gt_yaw, probas, coordinates, yaws)
            loss = distance_loss + yaw_loss + confidence_loss
            loss.backward()

            if "clip_grad_norm" in config["train"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
            optimizer.step()

            # log
            if num_steps % 1 == 0:
                pbar.set_description(f"epoch={epoch} loss={round(loss.item(), 2)} distance_loss={round(distance_loss.item(), 2)} yaw_loss={round(yaw_loss.item(), 2)} confidence_loss={round(confidence_loss.item(), 2)}")
            # validation
            if num_steps % config["train"]["validate_every_n_steps"] == 0 and num_steps > 0:
                del data
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    losses = []
                    for data in tqdm(val_dataloader):
                        if config["val"]["data_config"]["dataset_config"]["normlization"]:
                            data = normalize(data)
                        dict_to_cuda(data)
                        probas, coordinates, yaws = model(data, num_steps)
                        gt_xy = data["future/xy"] - data["history/xy"][:, :, -1:, :]
                        gt_yaw = data["future/yaw"] - data["history/yaw"][:, :, -1:, :]
                        gt_valid = mask_by_valid(data["future/valid"], data["agent_valid"])
                        distance_loss, yaw_loss, confidence_loss = loss_func(
                            gt_xy, gt_valid, gt_yaw, probas, coordinates, yaws)
                        loss = distance_loss + yaw_loss + confidence_loss
                        losses.append(loss.item())
                    pbar.set_description(f"validation loss = {round(sum(losses) / len(losses), 2)}")

                if sum(losses) / len(losses) < best_loss:
                    best_loss = sum(losses) / len(losses)
                    saving_data = {
                        "num_steps": num_steps,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    if config["train"]["scheduler"]:
                        saving_data["scheduler_state_dict"] = scheduler.state_dict()
                    torch.save(saving_data, os.path.join(args.save_folder, f"best_{epoch}.pth"))

            num_steps += 1
            if "max_iterations" in config["train"] and num_steps > config["train"]["max_iterations"]:
                break


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=False, default="configs/Multipathpp.yaml", help="Vectorizer Config")
    parser.add_argument("--save_folder", type=str, required=True, help="Save folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    train(args)