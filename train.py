# import datetime
import argparse
import os
import shutil

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import WrapperDecTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")
print(f"cuda total number of devices: {torch.cuda.device_count()}")


def get_lr_multiplier(step: int, cfg: DictConfig) -> float:
    """Get learning rate multiplier."""
    if step < cfg.train.warmup_steps:
        return (step + 1) / cfg.train.warmup_steps
    if step > cfg.train.decay_end_steps:
        return cfg.train.decay_end_multiplier
    position = (step - cfg.train.warmup_steps) / (cfg.train.decay_end_steps - cfg.train.warmup_steps)
    return 1 - (1 - cfg.train.decay_end_multiplier) * position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg_def = OmegaConf.load("configs/default.yaml")
    cfg_setting = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg_def, cfg_setting)

    # # wandb
    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg))

    # making save dir
    if cfg.save_model:
        # now_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(cfg.save_dir, cfg.wandb.name)
        os.makedirs(save_path, exist_ok=True)
        print("save_path:", save_path)

    #########################################################################
    #                            Load Dataset
    #########################################################################

    print("loading dataset...")
    train_dataset = CustomDataset(data_dir=cfg.data.data_dir, data_file=cfg.data.train_file)
    eval_dataset = CustomDataset(data_dir=cfg.data.data_dir, data_file=cfg.data.eval_file)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

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

    if torch.cuda.device_count() > 1:
        print("using DataParallel")
        model = torch.nn.DataParallel(model)

    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters: {n_parameters}")
    print(f"n_trainables: {n_trainables}")

    print("finished initializing model")

    #########################################################################
    #                               Training
    #########################################################################

    print("starting training...")

    if cfg.wandb.use:
        wandb.define_metric("step")
        wandb.define_metric("train/loss", step_metric="step")
        wandb.define_metric("train/lr", step_metric="step")
        wandb.define_metric("eval/loss", step_metric="step")

    optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: get_lr_multiplier(step, cfg),
    )
    optimizer.zero_grad()

    step = 0
    real_step = 0
    epoch = 0
    eval_loss_list = []
    not_improved_cnt = 0

    while real_step < cfg.train.max_steps and not_improved_cnt < cfg.train.early_stopping_patience:
        epoch += 1
        print("epoch:", epoch)
        model.train()
        recent_loss = []
        for batch in tqdm(train_loader, desc="train"):
            step += 1
            loss = model(batch.to(device))
            loss = loss.mean()
            recent_loss.append(float(loss))
            loss = loss / cfg.train.accumulation_steps
            loss.backward()

            if step % cfg.train.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                real_step += 1
                if cfg.wandb.use:
                    train_loss = np.mean(recent_loss)
                    recent_loss = []
                    wandb.log({"step": real_step, "train/loss": train_loss})
                    wandb.log({"step": real_step, "train/lr": scheduler.get_last_lr()[0]})

            # evaluation
            if step % (cfg.train.eval_steps * cfg.train.accumulation_steps) == 0 or real_step == cfg.train.max_steps:
                eval_loss = []
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc="eval", leave=False):
                        loss = model(batch.to(device))
                        loss = loss.mean()
                        eval_loss.append(float(loss))

                        del batch, loss

                if cfg.wandb.use:
                    wandb.log({"step": real_step, "eval/loss": np.mean(eval_loss)})
                else:
                    tqdm.write(f"step: {real_step}, eval/loss: {np.mean(eval_loss)}")
                eval_loss_list.append(np.mean(eval_loss))

                if len(eval_loss_list) == 1 or eval_loss_list[-1] < min(eval_loss_list[:-1]):
                    if cfg.save_model:
                        tqdm.write(f"step: {real_step}, eval loss improved, saving model...")
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), os.path.join(save_path, "model_best.pt"))
                        else:
                            torch.save(model.state_dict(), os.path.join(save_path, "model_best.pt"))
                    not_improved_cnt = 0
                else:
                    not_improved_cnt += 1
                    if not_improved_cnt >= cfg.train.early_stopping_patience:
                        tqdm.write(f"step: {real_step}, eval loss is not decreasing, stopping training...")
                        break
                if real_step == cfg.train.max_steps:
                    break
                model.train()

    print("finished training")


if __name__ == "__main__":
    main()
