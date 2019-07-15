from torch.utils.data import Dataset


class MoleculeDataset(Dataset):

    def __init__(self, train, validation, test, transform=None):
        super(MoleculeDataset, self).__init__()
        self.train = train
        self.train_molecules = list(train.keys())
        self.train_size = len(self.train_molecules)
        self.validation = validation
        self.validation_molecules = list(validation.keys())
        self.validation_size = len(self.validation_molecules)
        self.test = test
        self.test_molecules = list(test.keys())
        self.test_size = len(self.test_molecules)
        self.transform = transform

        self._lookup_dict = {"train": (self.train_molecules, self.train_size),
                             "validation": (self.validation_molecules, self.validation_size),
                             "test": (self.test_molecules, self.test_size)}

        self.set_split("train")

    def set_split(self, split="train"):
        self._target_split = split
        self._target, self._target_size = self._lookup_dict[split]

    def get_target(self):
        return self._target

    def __len__(self):
        return self._target_size

    def __getitem__(self, item):
        sample = self._target[item]
        if self.transform:
            sample = self.transform(sample)
        return sample
