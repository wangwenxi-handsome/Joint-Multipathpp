import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from prerender.features_description import generate_features_description
from prerender.configs import get_vectorizer_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument("--config", type=str, required=True, help="Vectorizer Config")
    parser.add_argument("--n-jobs", type=int, default=16, required=False, help="Number of threads")
    parser.add_argument("--num_parallel_reads", type=int, default=1, required=False, help="Number parallel of TFRecordDataset")
    args = parser.parse_args()
    return args

def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True

def render_and_save(vectorizer, data, output_path):
    data = tf.io.parse_single_example(data, generate_features_description())
    scene_data = vectorizer.render(data)
    scene_id = scene_data["scenario_id"]
    np.savez_compressed(os.path.join(output_path, f"scid_{scene_id}.npz"), **scene_data)

def main():
    args = parse_arguments()
    dataset = tf.data.TFRecordDataset(
        [os.path.join(args.datapath, f) for f in os.listdir(args.datapath)], num_parallel_reads=args.num_parallel_reads
    )
    vectorizer_config = get_vectorizer_config(args.config)
    vectorizer = vectorizer_config["class"](vectorizer_config)
    for data in tqdm(dataset.as_numpy_iterator()):
        data_to_numpy(data)
        render_and_save(vectorizer, data, args.output_path)


if __name__ == "__main__":
    t1 = time.time()
    main()
    print("prerender time cost:", time.time() - t1)
