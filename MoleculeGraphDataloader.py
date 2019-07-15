import numpy as np
import torch
from torch.utils.data import DataLoader
from .MoleculeGraphDataset import MoleculeGraphDataset, DataSet
from torchvision.transforms import Compose
import pandas as pd
from os import path
from .DataExtractor import *


class MoleculeGraphDataLoader:

    def __init__(self, num_valid, device="cpu"):
        self.num_valid = num_valid
        self.split = DataSet.TRAIN
        self.device = device
        self.train, self.validation, self.test = self.run_make_split()
        self.dataset = MoleculeGraphDataset(self.train, self.validation, self.test)
        self.set_data_set()

    def set_data_set(self, split=DataSet.TRAIN):
        self.split = split
        self.dataset.set_split(split)
        return self

    def generate_batches(self, batch_size, shuffle=True, drop_last=True):
        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=16, collate_fn=Compose([DataExtractor()]), pin_memory=False)
        for molecules in dataloader:
            yield molecules

    def run_make_split(self):
        split_dir = 'C:/Kaggle/Molecule/split'

        num_valid = self.num_valid
        train_dir = split_dir + '/train_split_by_mol.%d.npy'%num_valid
        valid_dir = split_dir + '/valid_split_by_mol.%d.npy' % num_valid
        test_dir = split_dir + '/test_split_by_mol.npy'

        if path.exists(train_dir) and path.exists(valid_dir) and path.exists(test_dir):
            train_split = np.load(train_dir)
            valid_split = np.load(valid_dir)
            test = np.load(test_dir)
        else:
            train_csv_file = 'C:/Kaggle/Molecule/train.csv'
            train_df = pd.read_csv(train_csv_file)
            train_molecule_names = train_df.molecule_name.unique()
            train_molecule_names = np.sort(train_molecule_names)

            np.random.shuffle(train_molecule_names)

            train_split = train_molecule_names[num_valid:]
            valid_split = train_molecule_names[:num_valid]

            test_csv_file = 'C:/Kaggle/Molecule/test.csv'
            test_df = pd.read_csv(test_csv_file)
            test = test_df.molecule_name.unique()

            np.save(train_dir, train_split)
            np.save(valid_dir, valid_split)
            np.save(test_dir, test)

        return train_split, valid_split, test

    def to_tensor(self, list_data, type="float"):
        if type == "float":
            return torch.from_numpy(np.array(list_data)).float().to(self.device, non_blocking=True)
        elif type == "int":
            return torch.from_numpy(np.array(list_data)).int().to(self.device, non_blocking=True)
