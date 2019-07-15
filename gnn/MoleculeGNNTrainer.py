from line_profiler import LineProfiler

from trainer import *
from gnn import MoleculeMPNN, MoleculeGraphDataLoader
from torch.nn.modules.loss import L1Loss
from mol import Struct
import numpy as np


class MoleculeGNNTrainer(BaseTrainer):

    def __init__(self, node_feature_size, edge_feature_size, node_hidden_size, edge_hidden_size,
                 num_epoch, batch_size, log_every=100, lr=0.01, dropout_ratio=0.5, steps=6):
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.steps = steps
        super(MoleculeGNNTrainer, self).__init__(num_epoch, batch_size, log_every, lr, dropout_ratio)
        self.update_tag()


    def create_model(self):
        dataloader = MoleculeGraphDataLoader(6000)
        model = MoleculeMPNN(self.node_feature_size, self.edge_feature_size, self.node_hidden_size,
                             self.edge_hidden_size, self.dropout_ratio, self.steps)
        loss = L1Loss()
        return dataloader, model, loss

    def to_tensor(self, batch: Struct):
        dict = {}
        new_batch = Struct()
        for key, value in batch.__dict__.items():
            if not isinstance(value, np.ndarray) and not isinstance(value, list):
                dict[key] = self.to_tensor_one(value)
            else:
                dict[key] = value
        new_batch = new_batch.from_dict(dict)
        return new_batch

    def get_response(self, graph):
        return graph.coupling_value

    def get_id(self, graph):
        return graph.id

    def set_node_hidden_size(self, node_hidden_size):
        self.node_hidden_size = node_hidden_size
        self.change_model()

    def set_edge_hidden_size(self, edge_hidden_size):
        self.edge_hidden_size = edge_hidden_size
        self.change_model()

    def set_steps(self, steps):
        self.steps = steps
        self.change_model()

    def set_dropout_ratio(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.change_model()

    def update_tag(self):
        self.tag += "nh" + str(self.node_hidden_size) + "eh" + str(self.edge_hidden_size) + "sp" + str(self.steps)
