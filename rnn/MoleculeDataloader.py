import numpy as np
import torch
from dataprocessor import DataProcessor
from torch.utils.data import DataLoader
from .MoleculeDataset import MoleculeDataset
from .MoleculeTensorMapper import MoleculeTensorMapper
from .MoleculePadder import MoleculePadder
from .ToTensor import ToTensor
from torchvision.transforms import Compose


class MoleculeDataLoader:

    def __init__(self, data_processor: DataProcessor, split_ratio, device="cpu"):
        self.data_processor = data_processor
        self.train, self.validation = self.data_processor.train_val_split(split_ratio)
        self.test = self.data_processor.test_series
        self.get_id = self.data_processor.get_train_id_by_molecule
        self.get_data_point = self.data_processor.get_train_data_point
        self.split = "train"
        self.device = device
        self.transformer = MoleculeTensorMapper(self.split, self.data_processor)
        self.set_data_set()

    def set_data_set(self, split="train"):
        self.split = split
        self.transformer.set_split(split)
        return self

    def generate_batches(self, batch_size, shuffle=True, drop_last=True):
        transformed_dataset = MoleculeDataset(self.train, self.validation, self.test,
                                              transform=Compose([self.transformer]))
        dataloader = DataLoader(dataset=transformed_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                num_workers=4, collate_fn=Compose([MoleculePadder()]), pin_memory=False)
        for molecules in dataloader:
            yield molecules

    def to_tensor(self, list_data, type="float"):
        if type == "float":
            return torch.from_numpy(np.array(list_data)).float().to(self.device, non_blocking=True)
        elif type == "int":
            return torch.from_numpy(np.array(list_data)).int().to(self.device, non_blocking=True)
