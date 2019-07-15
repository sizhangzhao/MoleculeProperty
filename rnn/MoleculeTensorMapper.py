from dataprocessor import DataProcessor


class MoleculeTensorMapper(object):

    def __init__(self, split, data_processor: DataProcessor):
        self.split = split
        self.data_processor = data_processor
        self.set_split()

    def set_split(self, split="train"):
        self.split = split
        if self.split == "test":
            self.get_id = self.data_processor.get_test_id_by_molecule
            self.get_data_point = self.data_processor.get_test_data_point
        else:
            self.get_id = self.data_processor.get_train_id_by_molecule
            self.get_data_point = self.data_processor.get_train_data_point
        return self

    def _get_data_by_molecule(self, molecule):
        ids = self.get_id(molecule)
        molecule_length = len(ids)
        single_molecule_feature = []
        single_molecule_response = []
        for single_id in ids:
            feature, response = self.get_data_point(single_id)
            single_molecule_feature.append(list(feature))
            single_molecule_response.append(response)

        return single_molecule_feature, single_molecule_response, molecule_length, ids

    def __call__(self, molecule):
        out_data = {}
        features, responses, length, all_ids = self._get_data_by_molecule(molecule)
        out_data['feature'] = features
        out_data['response'] = responses
        out_data['length'] = length
        out_data['id'] = all_ids
        return out_data
