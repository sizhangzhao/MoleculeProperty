import unittest
from gnn import MoleculeGraphDataLoader, MoleculeGNNTrainer
from gnn import DataSet


class TestModel(unittest.TestCase):

    def test_initial_Model(self):
        trainer = MoleculeGNNTrainer(3, 3, 3, 3, 3, 3)

    def test_model_train(self):
        trainer = MoleculeGNNTrainer(14, 5, 128, 128, 1, 16, 1)
        trainer.train()


if __name__ == '__main__':
    unittest.main()