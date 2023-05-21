import os
import glob
import random
import subprocess
import torch
import numpy as np
import yaml
from yaml import Loader


def set_random_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*.pth')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def dict_to_cuda(d):
    passing_keys = set([
        'history/lstm_data', 'history/lstm_data_diff',
        'history/mcg_input_data', 'history/mcg_input_data',
        'batch_size', 'road_network_embeddings', 
        'future/xy', 'future/valid', 'future/yaw', 
        "history/xy", "history/yaw", "yaw",
        'agent_valid', 'road_segments_valid'])
    for k in d.keys():
        if k not in passing_keys:
            continue
        v = d[k]
        if not isinstance(v, torch.Tensor):
            continue
        d[k] = d[k].cuda()


def get_yaml_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)
    return config


def mask_by_valid(input_tensor, mask):
    mask = mask.squeeze()
    return input_tensor * mask.reshape(*(
        *mask.shape, 
        *(1 for _ in range(len(input_tensor.shape) - len(mask.shape)))
    ))