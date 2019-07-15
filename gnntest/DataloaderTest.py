import unittest
from gnn import MoleculeGraphDataLoader
from gnn import DataSet


class TestDataLoader(unittest.TestCase):

    def test_dataloader(self):
        test_loader = MoleculeGraphDataLoader(6000)
        for molecule in test_loader.generate_batches(2):
            print(molecule)
            print(molecule.node.size())
            print(molecule.edge.size())
            break
        test_loader.set_data_set(DataSet.VAL)
        for molecule in test_loader.generate_batches(2):
            print(molecule)
            break
        test_loader.set_data_set(DataSet.TEST)
        for molecule in test_loader.generate_batches(2, shuffle=False, drop_last=False):
            print(molecule)
            break


if __name__ == '__main__':
    unittest.main()