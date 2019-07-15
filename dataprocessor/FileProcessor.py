import pandas as pd
import configparser
import os


class FileProcessor(object):

    def __init__(self, stucture_only=True):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.stucture_only = stucture_only
        self.structure_merge_train_name = self.config['FILES']['structure_only_train']
        self.structure_merge_test_name = self.config['FILES']['structure_only_test']
        self.structure_merge_train_exist = os.path.isfile(self.structure_merge_train_name)
        self.structure_merge_test_exist = os.path.isfile(self.structure_merge_test_name)
        if not self.structure_merge_train_exist:
            self.train = pd.read_csv(self.config['FILES']['train'])
        if not self.structure_merge_test_exist:
            self.test = pd.read_csv(self.config['FILES']['test'])
        if not self.structure_merge_train_exist or not self.structure_merge_test_exist:
            self.structure = pd.read_csv(self.config['FILES']['structure'])
        if not self.stucture_only:
            self.dipole = pd.read_csv(self.config['FILES']['dipole'])
            self.tensors = pd.read_csv(self.config['FILES']['tensors'])
            self.mulliken = pd.read_csv(self.config['FILES']['mulliken'])
            self.energy = pd.read_csv(self.config['FILES']['energy'])
            self.separate = pd.read_csv(self.config['FILES']['separate'])

    def get_stucture(self):
        if self.structure_merge_train_exist:
            merged_structure_train = pd.read_csv(self.structure_merge_train_name)
        else:
            merged_structure_train = \
                self.train.pipe(pd.merge, self.structure, how='left',
                                left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index"])\
                    .pipe(pd.merge, self.structure, how='left',
                          left_on=["molecule_name", "atom_index_1"], right_on=["molecule_name", "atom_index"], suffixes=["_0", "_1"])
            merged_structure_train.to_csv(self.structure_merge_train_name, index=False)
        if self.structure_merge_test_exist:
            merged_structure_test = pd.read_csv(self.structure_merge_test_name)
        else:
            merged_structure_test = \
                self.test.pipe(pd.merge, self.structure, how='left',
                               left_on=["molecule_name", "atom_index_0"], right_on=["molecule_name", "atom_index"])\
                    .pipe(pd.merge, self.structure, how='left',
                          left_on=["molecule_name", "atom_index_1"], right_on=["molecule_name", "atom_index"], suffixes=["_0", "_1"])
            merged_structure_test.to_csv(self.structure_merge_test_name, index=False)
        return merged_structure_train, merged_structure_test


if __name__ == '__main__':
    files = FileProcessor()
    train, test = files.get_stucture()