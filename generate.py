import argparse
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import WrapperDecTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="continuation")
    parser.add_argument("--initial_length", type=int, default=10)
    parser.add_argument("--scratch_num", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_def = OmegaConf.load("configs/default.yaml")
    cfg_setting = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)

    if args.mode == "continuation":
        gen_save_path = os.path.join("example", "continuation", cfg.wandb.name)
        tgt_save_path = os.path.join("example", "continuation", "test_data")
        os.makedirs(tgt_save_path, exist_ok=True)
    elif args.mode == "scratch":
        gen_save_path = os.path.join("example", "scratch", cfg.wandb.name)

    os.makedirs(gen_save_path, exist_ok=True)

    #########################################################################
    #                            Load Dataset
    #########################################################################

    test_dataset = CustomDataset(data_dir=cfg.data.data_dir, data_file=cfg.data.test_file)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)
    print("finished loading dataset")

    #########################################################################
    #                               Model
    #########################################################################

    print("initializing model...")
    model = WrapperDecTransformer(
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        num_tokens=cfg.model.num_tokens,
        max_seq_len=cfg.model.max_seq_len,
        heads=cfg.model.heads,
        emb_dropout=cfg.model.dropout,
        attn_dropout=cfg.model.dropout,
        ff_dropout=cfg.model.dropout,
        ff_inner_dim=cfg.model.ff_inner_dim,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.save_dir, cfg.wandb.name, "model_best.pt")))
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters: {n_parameters}")
    print(f"n_trainables: {n_trainables}")

    print("finished loading model")

    #########################################################################
    #                            Generation
    #########################################################################

    with torch.no_grad():
        print("starting generation...")
        # generation
        if args.mode == "continuation":
            for idx, batch in tqdm(enumerate(test_loader)):
                prompt = batch[:, : args.initial_length].to(device)
                gen_seq = model.generate(prompt, seq_len=cfg.model.max_seq_len, eos_token=2)
                gen_seq = gen_seq[0].cpu().tolist()
                with open(os.path.join(gen_save_path, f"{idx}.txt"), "w") as f:
                    f.write(str(gen_seq))
                with open(os.path.join(tgt_save_path, f"{idx}.txt"), "w") as f:
                    f.write(str(batch[0].tolist()))
        elif args.mode == "scratch":
            prompt = torch.zeros(1, 1, dtype=torch.long).to(device)
            for idx in tqdm(range(args.scratch_num)):
                gen_seq = model.generate(prompt, seq_len=cfg.model.max_seq_len, eos_token=2)
                gen_seq = gen_seq[0].cpu().tolist()
                with open(os.path.join(gen_save_path, f"{idx}.txt"), "w") as f:
                    f.write(str(gen_seq))

    print("finished generation")


if __name__ == "__main__":
    main()
