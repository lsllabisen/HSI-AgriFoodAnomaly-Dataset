from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class HSIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.cache = {}  # Dictionnaire pour cache

        self.label_map = {'normal': 0, 'anormal': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_dat = row['path_dat']
        path_npy = row['path_npy']
        index_in_tensor = int(row['index'])
        label = int(row['label'])


        patches_shape = np.load(path_npy)
        patches_shape = tuple(patches_shape)
        patches_memmap = np.memmap(path_dat, dtype='float32', mode='r', shape=patches_shape)

        tensor = patches_memmap[index_in_tensor]  # shape: (1, bands, lines, patch)

        if self.transform:
            tensor = self.transform(tensor)

        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)