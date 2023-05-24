"""
The results predicted by the model are stored as the pth file.
Each pth has many scenarios, which costs a lof of time when reading.
So This code turns the pth file to several npz files, each npz file corresponding to one sceanrio, which can be used for submit.py or submit4motion.py.
"""
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import shutil


filenames = glob("/root/competition/dataset/output/testing32/*")
dst_dir = "/root/competition/dataset/output/testing32_decompress/"

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)

for shard_filename in tqdm(filenames):
    data_dict = torch.load(shard_filename, map_location='cpu')
    for sc_id in data_dict.keys():
        states = np.concatenate((data_dict[sc_id]["xyz"].numpy(), data_dict[sc_id]["yaws"].numpy()), -1)
        agent_id =  data_dict[sc_id]["agent_id"].numpy().astype(np.int32)
        if data_dict[sc_id].get("probs") is not None:
            confidence = data_dict[sc_id]["probs"].numpy()
            np.savez(os.path.join(dst_dir, f"scid_{sc_id}"), states=states, agent_id=agent_id, confidence=confidence)
        else:
            np.savez(os.path.join(dst_dir, f"scid_{sc_id}"), states=states, agent_id=agent_id)
    os.remove(shard_filename)
