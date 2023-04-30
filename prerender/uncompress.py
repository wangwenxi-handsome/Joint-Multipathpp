import os
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_data_path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--uncompress_data_path", type=str, required=True, help="Path to save data")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    files = [os.path.join(args.compress_data_path, f) for f in os.listdir(args.compress_data_path)]
    for f in files:
        np_data = dict(np.load(f, allow_pickle=True))
        for k in np_data:
            np.savez(os.path.join(args.uncompress_data_path, k), np_data[k])


if __name__ == "__main__":
    main()