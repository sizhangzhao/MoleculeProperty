import torch
import numpy as np


class ToTensor(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, out_data):
        final_out_data = {}
        final_out_data["feature"] = self.to_tensor(out_data["feature"])
        final_out_data["response"] = self.to_tensor(out_data["response"])
        final_out_data["length"] = out_data["length"]
        final_out_data["id"] = out_data["id"]
        return final_out_data

    def to_tensor(self, list_data):
        return torch.from_numpy(np.array(list_data)).float().to(self.device)
