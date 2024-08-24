from typing import Union

import numpy as np
from torch.utils.data import DataLoader, random_split

from .dataset import MzmlDataset

class Loader:
    def __init__(self, split_ratio: float, task: Union[MzmlDataset], batch_size: int = 16, n_works: int = 4) -> None:
        self.task = task
        self.n_works = n_works
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def read_data(self, data_path: str):
        return np.load(data_path, allow_pickle=True)
    
    def dataset(self, data_path: str):
        data = self.read_data(data_path)
        return self.task(data), data['Info']
    
    def train_loader(self, data_path: str):
        dataset, _ = self.dataset(data_path)
        validate_length = int(len(dataset) * self.split_ratio)
        train_data, validate_data = random_split(
        dataset, [len(dataset) - validate_length, validate_length])
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.n_works), \
        DataLoader(validate_data, batch_size=self.batch_size,
                   shuffle=True, num_workers=self.n_works)

    def test_loader(self, data_path: str):
        dataset, info = self.dataset(data_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_works), info