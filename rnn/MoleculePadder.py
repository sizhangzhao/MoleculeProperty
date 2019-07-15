from typing import List, Dict
import torch
import numpy as np


class MoleculePadder(object):

    def __call__(self, molecules: List[Dict]):
        out_data = {}
        features = []
        responses = []
        all_ids = []
        lengths = []
        molecules = sorted(molecules, key=lambda molecule: molecule["length"], reverse=True)
        longest = molecules[0]["length"]
        for molecule in molecules:
            curr_length = molecule["length"]
            lengths.append(curr_length)
            num_pad = longest - curr_length
            single_molecule_feature = molecule["feature"]
            single_molecule_response = molecule["response"]
            single_molecule_id = molecule["id"]
            num_feature = len(single_molecule_feature[0])
            single_molecule_feature += [[0 for _ in range(num_feature)] for _ in range(num_pad)]
            single_molecule_response += [0 for _ in range(num_pad)]
            single_molecule_id += [-1 for _ in range(num_pad)]
            features.append(single_molecule_feature)
            responses.append(single_molecule_response)
            all_ids.append(single_molecule_id)
        out_data["feature"] = features
        out_data["response"] = responses
        out_data["length"] = lengths
        out_data["id"] = all_ids
        return out_data
