from typing import List, Tuple, Dict
from rnn import MoleculeRNNRunner
import pandas as pd
import time


class HyperParameterTuner:

    def __init__(self, parameters: Dict, rnn_runner: MoleculeRNNRunner):
        self.parameters = parameters
        self.rnn_runner = rnn_runner
        self.generate_parameter_setter_mapping()
        self.tune_loss = {"tag": [], "train_loss": [], "val_loss": []}

    def generate_parameter_setter_mapping(self):
        self.parameter_setter_mapping = {"hidden_size": self.rnn_runner.set_hidden_size,
                                    "dropout_ratio": self.rnn_runner.set_dropout_ratio,
                                    "attention": self.rnn_runner.set_attention,
                                    "lr": self.rnn_runner.set_lr,
                                    "batch_size": self.rnn_runner.set_batch_size}

    def tune(self):
        for parameter_name in self.parameters.keys():
            setter = self.parameter_setter_mapping[parameter_name]
            for parameter_value in self.parameters[parameter_name]:
                print("Tuning for " + parameter_name + " for value " + str(parameter_value))
                setter(parameter_value)
                val_loss: float
                loss, val_loss = self.rnn_runner.train()
                self.tune_loss["tag"].append(self.rnn_runner.tag)
                self.tune_loss["train_loss"].append(loss)
                self.tune_loss["val_loss"].append(val_loss)
        tune_df = pd.DataFrame.from_dict(self.tune_loss)
        tune_df.to_csv("tune/" + str(time.time()) + ".csv", index=False)
