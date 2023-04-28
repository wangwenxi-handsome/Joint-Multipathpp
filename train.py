import os
import sys
import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
from prerender.utils.utils import data_to_numpy, get_config
import subprocess
from matplotlib import pyplot as plt
from utils.utils import set_random_seed, get_last_file, get_git_revision_short_hash, dict_to_cuda


def train(args):
    # init
    set_random_seed(42)
    config = get_config(args.config)
    models_path = os.path.join("models", get_git_revision_short_hash())
    if(~os.path.exists(models_path)):
        os.mkdir(models_path)
    last_checkpoint = get_last_file(models_path)
    dataloader = get_dataloader(config["train"]["data_config"])
    val_dataloader = get_dataloader(config["val"]["data_config"])
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
    this_num_steps = 0
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("N PARAMS=", params)

    # train and validation
    train_losses = []
    for _ in tqdm(range(config["train"]["n_epochs"])):
        pbar = tqdm(dataloader)
        for data in pbar:
            # train
            model.train()
            optimizer.zero_grad()
            if config["train"]["normalize"]:
                data = normalize(data, config)
            dict_to_cuda(data)
            probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)
            assert torch.isfinite(coordinates).all()
            assert torch.isfinite(probas).all()
            assert torch.isfinite(covariance_matrices).all()

            # loss and optimizer
            xy_future_gt = data["future/xy"]
            if config["train"]["normalize_output"]:
                # assert not (config["train"]["normalize_output"] and config["train"]["trainable_cov"])
                xy_future_gt = (data["future/xy"] - torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()) / 10.
                assert torch.isfinite(xy_future_gt).all()
                _coordinates = coordinates.detach() * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
            else:
                _coordinates = coordinates.detach()
            loss = nll_with_covariances(
                xy_future_gt, coordinates, probas, data["future/valid"].squeeze(-1),
                covariance_matrices) * loss_coeff
            train_losses.append(loss.item())
            loss.backward()
            if "clip_grad_norm" in config["train"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
            optimizer.step()

            # log
            if num_steps % 10 == 0:
                pbar.set_description(f"loss = {round(loss.item(), 2)}")
            # save
            if num_steps % 1000 == 0 and this_num_steps > 0:
                saving_data = {
                    "num_steps": num_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if config["train"]["scheduler"]:
                    saving_data["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(saving_data, os.path.join(models_path, f"last.pth"))
            # validation
            if num_steps % (len(dataloader) // 2) == 0 and this_num_steps > 0:
                del data
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    losses = []
                    min_ades = []
                    first_batch = True
                    for data in tqdm(val_dataloader):
                        if config["train"]["normalize"]:
                            data = normalize(data, config)
                        dict_to_cuda(data)
                        probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)
                        if config["train"]["normalize_output"]:
                            xy_future_gt = (data["future/xy"] - torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()) / 10.
                            assert torch.isfinite(xy_future_gt).all()
                            coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
                        loss = nll_with_covariances(
                            xy_future_gt, coordinates, probas, data["future/valid"].squeeze(-1),
                            covariance_matrices) * loss_coeff
                        losses.append(loss.item())
                    pbar.set_description(f"validation loss = {round(sum(losses) / len(losses), 2)}")
                    train_losses = []
                
                saving_data = {
                    "num_steps": num_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if config["train"]["scheduler"]:
                    saving_data["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(saving_data, os.path.join(models_path, f"{num_steps}.pth"))

            num_steps += 1
            this_num_steps += 1
            if "max_iterations" in config["train"] and num_steps > config["train"]["max_iterations"]:
                break


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--validation-data-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Vectorizer Config")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    train(args)