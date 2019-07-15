from torch.utils.data import Dataset
from mol import read_pickle_from_file, GRAPH_DIR
from trainer import DataSet


class MoleculeGraphDataset(Dataset):

    def __init__(self, train, validation, test):
        super(MoleculeGraphDataset, self).__init__()
        self.train = train
        self.train_size = len(self.train)
        self.validation = validation
        self.validation_size = len(self.validation)
        self.test = test
        self.test_size = len(self.test)

        self._lookup_dict = {DataSet.TRAIN: (self.train, self.train_size),
                             DataSet.VAL: (self.validation, self.validation_size),
                             DataSet.TEST: (self.test, self.test_size)}

        self.set_split(DataSet.TRAIN)

    def set_split(self, split=DataSet.TRAIN):
        self._target_split = split
        self._target, self._target_size = self._lookup_dict[split]

    def get_target(self):
        return self._target

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        molecule_name = self._target[index]
        graph_file = GRAPH_DIR + '/%s.pickle' % molecule_name
        graph = read_pickle_from_file(graph_file)
        assert (graph.molecule_name == molecule_name)
        return graph
