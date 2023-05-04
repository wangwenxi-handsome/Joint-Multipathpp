import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress_data_path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--uncompress_data_path", type=str, required=True, help="Path to save data")
    parser.add_argument("--n_jobs", type=int, default=0, required=False, help="Number of processes")
    args = parser.parse_args()
    return args


def uncompress_one_file(file, data_path):
    np_data = dict(np.load(file, allow_pickle=True))
    for k in np_data:
        np.savez(os.path.join(data_path, k), **np_data[k])

def main():
    args = parse_arguments()
    files = [os.path.join(args.compress_data_path, f) for f in os.listdir(args.compress_data_path)]
    if args.n_jobs == 0:
        for f in tqdm(files):
            uncompress_one_file(f, args.uncompress_data_path)
    else:
        pool = multiprocessing.Pool(args.n_jobs)
        processes = []
        for f in files:
            processes.append(pool.apply_async(uncompress_one_file, args=(f, args.uncompress_data_path)))
        for p in tqdm(processes):
            p.get()


if __name__ == "__main__":
    main()