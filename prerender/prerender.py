import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import multiprocessing
from features_description import generate_features_description
from configs import get_vectorizer_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save data")
    parser.add_argument("--config", type=str, required=True, help="Vectorizer Config")
    parser.add_argument("--n_jobs", type=int, default=0, required=False, help="Number of processes")
    args = parser.parse_args()
    return args

def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True

def render_and_save_one_tfrecord(file, vectorizer, output_path):
    dataset = tf.data.TFRecordDataset([file])
    for data in dataset.as_numpy_iterator():
        data = tf.io.parse_single_example(data, generate_features_description())
        data_to_numpy(data)
        scene_data = vectorizer.render(data, os.path.basename(file))
        scene_id = scene_data["scenario_id"]
        np.savez_compressed(os.path.join(output_path, f"scid_{scene_id}.npz"), **scene_data)

def main():
    args = parse_arguments()
    files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path)]
    vectorizer_config = get_vectorizer_config(args.config)
    vectorizer = vectorizer_config["class"](vectorizer_config)

    if args.n_jobs == 0:
        for file in tqdm(files):
            render_and_save_one_tfrecord(file, vectorizer, args.output_path)
    else:
        pool = multiprocessing.Pool(args.n_jobs)
        processes = []
        for file in files:
            processes.append(pool.apply_async(render_and_save_one_tfrecord, args=(file, vectorizer, args.output_path)))
        for p in tqdm(processes):
            p.get()


if __name__ == "__main__":
    main()
