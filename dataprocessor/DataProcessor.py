import pandas as pd
import math
import random


class Config(object):
    distance_colume_name = "distance"
    x1 = "x_0"
    x2 = "x_1"
    y1 = "y_0"
    y2 = "y_1"
    z1 = "z_0"
    z2 = "z_1"
    train_file_name = "structure_only_train_after_process.csv"
    test_file_name = "structure_only_test_after_process.csv"
    cat_columns = ["type", "atom_0", "atom_1"]
    molecule = "molecule_name"
    id = "id"
    indices = [id]
    columns = 'scalar_coupling_constant|id|molecule_name|^type_|^atom_0_|^atom_1_|distance|^x_|^y_|^z_'
    y = "scalar_coupling_constant"
    x = '^type_|^atom_0_|^atom_1_|distance|^x_|^y_|^z_'


class DataProcessor(object):
    config = Config()

    def __init__(self, train=None, test=None):
        self.train = train
        self.test = test
        self.train_series = None
        self.test_series = None

    def load_dataset(self, nrows=None):
        self.train = pd.read_csv(self.config.train_file_name, nrows=nrows)
        self.test = pd.read_csv(self.config.test_file_name, nrows=nrows)
        return self.get_molecule_series().reset_index()

    def __calculate_distance(self, row):
        return math.sqrt(math.pow((row[self.config.x2] - row[self.config.x1]), 2) +
                         math.pow((row[self.config.y2] - row[self.config.y1]), 2) +
                         math.pow((row[self.config.z2] - row[self.config.z1]), 2))

    def add_distance(self):
        self.train[self.config.distance_colume_name] = self.train.apply(lambda row: self.__calculate_distance(row), axis=1)
        self.test[self.config.distance_colume_name] = self.test.apply(lambda row: self.__calculate_distance(row), axis=1)
        return self

    def get_feature_size(self):
        return len(self.train.iloc[0].filter(regex=self.config.x))

    @staticmethod
    def get_one_hot_encode(data, column_name):
        return pd.get_dummies(data[column_name], prefix=column_name)

    def cat_encode(self):
        for column in self.config.cat_columns:
            self.train = pd.concat([self.train, self.get_one_hot_encode(self.train, column)], axis=1)
            self.test = pd.concat([self.test, self.get_one_hot_encode(self.test, column)], axis=1)
        return self

    def filter(self):
        self.train = self.train.filter(regex=self.config.columns)
        self.test = self.test.filter(regex=self.config.columns)
        return self

    def save_dataset(self):
        self.train.to_csv(self.config.train_file_name, index=False)
        self.test.to_csv(self.config.test_file_name, index=False)
        return self

    def get_molecule_series(self):
        self.train_series = self.train.groupby(self.config.molecule)[self.config.id].apply(list).to_dict()
        self.test_series = self.test.groupby(self.config.molecule)[self.config.id].apply(list).to_dict()
        return self

    def train_val_split(self, split_ratio):
        molecules = list(self.train_series.keys())
        n_total = len(molecules)
        n_train = int(n_total * split_ratio)
        shuffled_molecule = random.sample(molecules, n_total)
        shuffled_train = dict()
        validation = dict()
        for molecule in shuffled_molecule[:n_train]:
            shuffled_train[molecule] = self.train_series[molecule]
        for molecule in shuffled_molecule[n_train:]:
            validation[molecule] = self.train_series[molecule]
        return shuffled_train, validation

    def reset_index(self):
        self.train.set_index(self.config.indices, inplace=True)
        self.test.set_index(self.config.indices, inplace=True)
        return self

    def get_train_id_by_molecule(self, molecule):
        return self.train_series[molecule]

    def get_test_id_by_molecule(self, molecule):
        return self.test_series[molecule]

    def get_train_data_point(self, _id):
        return self.train.loc[_id].filter(regex=self.config.x), self.train.loc[_id][self.config.y]

    def get_test_data_point(self, _id):
        return self.test.loc[_id].filter(regex=self.config.x), None

    def get_dataset(self):
        return self.train, self.test