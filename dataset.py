import os

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, data_file):
        data_path = os.path.join(data_dir, data_file)
        self.dataset = torch.load(data_path)
        self.data_len = self.dataset.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample
