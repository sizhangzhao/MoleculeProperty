from typing import List, Dict
import torch
import numpy as np
from mol import Struct


class DataExtractor(object):
    """
    node: (\sum num_atom) * num_node_feature
    edge: (\sum num_edge) * num_edge_feature
    edge_index: (\sum num_edge) * 2  (num_atom * num_atom - num_atom)
    node_batch_index: (\sum num_atom)
    """

    def __call__(self, batch):

        batch_size = len(batch)

        node = []
        edge = []
        edge_index = []
        node_batch_index = []

        coupling_id = []
        coupling_index = []
        coupling_type = []
        coupling_type_back = []
        coupling_type_index = []
        coupling_value = []
        coupling_batch_index = []
        infor = []

        offset = 0
        for b in range(batch_size):
            graph = batch[b]
            # print(graph.molecule_name)

            num_node = len(graph.node)
            node.append(graph.node)
            edge.append(graph.edge)
            edge_index.append(graph.edge_index + offset)
            node_batch_index.append([b] * num_node)

            num_coupling = len(graph.coupling.value)
            coupling_id.append(graph.coupling.id)
            coupling_index.append(graph.coupling.index + offset)
            coupling_type.append(graph.coupling.type)
            coupling_type_back.append(graph.coupling.type_backup)
            coupling_type_index.append(graph.coupling.type_backup)
            coupling_value.append(graph.coupling.value)
            coupling_batch_index.append([b] * num_coupling)

            infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
            offset += num_node

        node = torch.from_numpy(np.concatenate(node)).float()
        edge = torch.from_numpy(np.concatenate(edge)).float()
        edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
        node_batch_index = torch.from_numpy(np.concatenate(node_batch_index)).long()

        coupling_index = torch.from_numpy(np.concatenate(coupling_index)).long()
        coupling_type = torch.from_numpy(np.concatenate(coupling_type)).long()
        coupling_type_back = torch.from_numpy(np.concatenate(coupling_type_back)).long()
        coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
        coupling_batch_index = torch.from_numpy(np.concatenate(coupling_batch_index)).long()
        coupling_id = np.concatenate(coupling_id).tolist()
        return Struct(
            node=node,
            edge=edge,
            edge_index=edge_index,
            node_batch_index=node_batch_index,
            coupling_index=coupling_index,
            coupling_type=coupling_type,
            coupling_type_back=coupling_type_back,
            coupling_value=coupling_value,
            coupling_batch_index=coupling_batch_index,
            id=coupling_id,
            infor=infor
        )
