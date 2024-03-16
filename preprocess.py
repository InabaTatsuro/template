import argparse
import os
import random

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = torch.randint(0, 10000, (100, 1024))
    for i in range(100):
        rand_idx = random.randint(1, 1023)
        dataset[i, 0] = 1  # start token
        dataset[i, rand_idx] = 2  # end token
        dataset[i, rand_idx + 1 :] = -100  # ignore token

    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    eval_len = int(data_len * 0.1)
    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len : train_len + eval_len]
    test_dataset = dataset[train_len + eval_len :]
    torch.save(train_dataset, os.path.join(args.out_dir, "train.pt"))
    torch.save(val_dataset, os.path.join(args.out_dir, "eval.pt"))
    torch.save(test_dataset, os.path.join(args.out_dir, "test.pt"))
    print("done")


if __name__ == "__main__":
    main()
