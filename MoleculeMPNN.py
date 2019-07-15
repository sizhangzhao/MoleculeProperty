import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch
import torch_geometric.nn as gnn
from mol import Struct
from typing import Callable
import numpy as np
import torch.nn.functional as F


#https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972#latest-573627
class LinearBlock(nn.Module):

    def __init__(self, input_size, output_size, dropout_ratio=0., bias=True, act=None):
        super(LinearBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size, bias=bias)
            # , nn.Dropout(dropout_ratio)
        )
        self.act = act

    def forward(self, x):
        x = self.model(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MoleculeMPNN(nn.Module):

    def __init__(self, node_feature_size, edge_feature_size, node_hidden_size, edge_hidden_size,
                 dropout_ratio=0.5, steps=6):
        super(MoleculeMPNN, self).__init__()
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.node_hidden_size = node_hidden_size
        self.edge_feature_size = edge_hidden_size
        self.dropout_ratio = dropout_ratio

        self.embedder = nn.Sequential(
            LinearBlock(node_feature_size, 64, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(64, self.node_hidden_size, self.dropout_ratio, False),
        )

        self.steps = steps

        self.edge_net = nn.Sequential(
            LinearBlock(edge_feature_size, 32, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(32, 64, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(64, self.edge_feature_size, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(self.edge_feature_size, self.node_hidden_size * self.node_hidden_size, self.dropout_ratio, True)
        )

        self.mpnn = gnn.NNConv(self.node_hidden_size, self.node_hidden_size,
                               self.edge_net, aggr="mean", root_weight=True)

        self.gru = nn.GRUCell(self.node_hidden_size, self.node_hidden_size)

        self.set2set = gnn.Set2Set(self.node_hidden_size, self.steps)

        self.fc = nn.Sequential(
            LinearBlock(self.node_hidden_size * 4 + 8, 1024, self.dropout_ratio, True, nn.ReLU()),
            LinearBlock(1024, 8)
        )

    def forward(self, graph: Struct):
        node, edge, edge_index, node_batch_index, coupling_index, coupling_type, coupling_type_back,\
            coupling_value, coupling_batch_index = self.get_graph(graph)

        edge_index = edge_index.t().contiguous()

        node_embedding = F.relu(self.embedder(node)) # num_atom * node_hidden_size
        h = node_embedding
        accum_node = node_embedding

        for i in range(self.steps):
            accum_node = F.relu(self.mpnn(accum_node, edge_index, edge)) # num_atom * node_hidden_size
            accum_node = self.gru(accum_node, h) # num_atom * node_hidden_size
            h = accum_node

        torch.cuda.empty_cache()

        pool: torch.tensor = self.set2set(accum_node, node_batch_index) # num_mol * (2 * node_hidden_size)

        pool = pool.index_select(
            dim=0,
            index=coupling_batch_index
        ) # num_coupling * (2 * node_hidden_size)

        node_feature = accum_node.index_select(
            dim=0,
            index=coupling_index.view(-1)
        ).reshape(len(coupling_index), -1) # num_coupling * (2 * node_hidden_size)

        # edges = self.find_edge(edge, edge_index, coupling_index)

        features = torch.cat([pool, node_feature, coupling_type.float()], -1)

        predictions = self.fc(features)

        prediction = torch.gather(predictions, 1, coupling_type_back.view(-1, 1)).view(-1)

        return prediction

    #https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
    def find_edge(self, edge: torch.Tensor, edge_index: torch.Tensor, coupling_index: torch.Tensor) -> torch.Tensor:
        sorted_edge = self.apply_axis(self.encode_edge, edge_index, dim=1).numpy()
        sorted_coupling = self.apply_axis(self.encode_edge, coupling_index, dim=0).numpy()
        coupling_edge_index = [torch.Tensor(np.where(sorted_edge == m)[0][0]) for m in np.nditer(sorted_coupling)]
        coupling_edge_index = torch.stack(coupling_edge_index, dim=-1).cuda()
        return edge.index_select(dim=0, index=coupling_edge_index)

    @staticmethod
    def encode_edge(edge: torch.Tensor):
        sorted_edge, _ = edge.view(-1).sort(-1)
        encoded_edge = "".join([str(m.item()) for m in torch.unbind(sorted_edge, dim=0)])
        return torch.tensor([hash(encoded_edge)], dtype=torch.int64)

    @staticmethod
    def apply_axis(func: Callable, tensor: torch.Tensor, dim=0) -> torch.Tensor:
        tList = [func(m) for m in torch.unbind(tensor, dim=dim)]
        res = torch.stack(tList, dim=dim)
        return res.view(-1)

    @staticmethod
    def get_graph(graph):
        return graph.node, graph.edge, graph.edge_index, graph.node_batch_index, graph.coupling_index, \
               graph.coupling_type, graph.coupling_type_back, graph.coupling_value, graph.coupling_batch_index
