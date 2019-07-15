from line_profiler import LineProfiler

from .MoleculeRNN import *
from .MoleculeDataloader import *
from .PaddedL1Loss import *
from .MoleculeSeq2Seq import *
from .MoleculeAtt import *
from dataprocessor import *
from torch.nn.modules.loss import L1Loss
import time
import pandas as pd
from tensorboardX import SummaryWriter
import math
import matplotlib.pyplot as plt
import line_profiler


class MoleculeRNNTrainer:
    profile: LineProfiler = line_profiler.LineProfiler()

    def __init__(self, dataprocessor: DataProcessor, split_ratio, num_epoch, hidden_size, clip_grad=0.5, lr=0.01,
                 dropout_ratio=0.5, log_every=100, batch_size=16, use_attention=True, evaluate_val_each_step=False):

        self.dataprocessor = dataprocessor
        self.split_ratio = split_ratio
        self.feature_size = self.dataprocessor.get_feature_size()
        self.num_epoch = num_epoch
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.lr = lr
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataloader = MoleculeDataLoader(self.dataprocessor, self.split_ratio, self.device)
        self.clip_grad = clip_grad
        self.log_every = log_every
        self.batch_size = batch_size
        self.use_attention = use_attention
        self.model = self.create_model()
        self.move_device()
        self.loss_function = PaddedL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.writer = SummaryWriter('log')
        self.evaluate_val_each_step = evaluate_val_each_step
        self.generate_basic_model_information()

    def create_model(self):
        rnnmodel = MoleculeRNN(self.feature_size, self.hidden_size, self.dropout_ratio, self.use_attention, self.device)
        stsmodel = MoleculeSeq2Seq(6, self.feature_size, self.hidden_size, self.device, self.dropout_ratio)
        attmodel = MoleculeAtt(6, self.feature_size, self.hidden_size, self.device, self.dropout_ratio)
        return attmodel

    def generate_basic_model_information(self):
        self.a = "a" if self.use_attention else "na"
        self.tag = "s" + str(self.split_ratio) + "e" + str(self.num_epoch) + "h" + str(self.hidden_size) + "c" + \
                   str(self.clip_grad) + "lr" + str(self.lr) + "d" + str(self.dropout_ratio) + "b" \
                   + str(self.batch_size) + self.a
        self.num_iter = 0

    def change_model(self):
        self.model = self.create_model()
        self.move_device()
        self.generate_basic_model_information()

    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
        self.change_model()

    def set_dropout_ratio(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.change_model()

    def set_attention(self, use_attention):
        self.use_attention = use_attention
        self.change_model()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.change_model()

    def set_lr(self, lr):
        self.lr = lr
        self.change_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def move_device(self):
        # if torch.cuda.device_count() > 1 and self.device != "cpu":
        #     self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    # @profile
    def train(self):
        self.dataloader.set_data_set("train")
        model = self.model
        writer = self.writer
        losses = {"num_iter": [], "loss": []}
        val_losses = {"num_iter": [], "loss": []}
        model.train()
        loss_function = self.loss_function
        optimizer = self.optimizer
        num_iter = self.num_iter
        start_time = time.time()
        total_loss = 0
        avg_loss = 0
        val_loss = 0
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            batch_generator = self.dataloader.generate_batches(self.batch_size)
            for batch_index, batch_molecule in enumerate(batch_generator):
                num_iter += 1
                optimizer.zero_grad()
                features = self.dataloader.to_tensor(batch_molecule['feature'])
                response = self.dataloader.to_tensor(batch_molecule['response'])
                length = batch_molecule['length']
                y_pred = model(features, length)
                loss = loss_function(y_pred, response, length)
                loss_batch = loss.item()
                epoch_loss += loss_batch
                total_loss += loss_batch
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()

                if num_iter % self.log_every == 0:
                    avg_loss = total_loss / (num_iter - self.num_iter)
                    print("Training: epoch %d, iter %d, epoch_loss: %.2f, total_loss: %.2f, time: %.2f"
                          % (epoch, num_iter, epoch_loss, avg_loss, time.time() - start_time))
                    # print("y_pred at epoch: " + str(epoch) + str(y_pred))
                    # print("response at epoch: " + str(epoch) + str(response))
                    writer.add_histogram(self.tag + "/Train/prediction", y_pred, num_iter)
                    writer.add_histogram(self.tag + "/Train/true", response, num_iter)
                    writer.add_scalar(self.tag + '/Train/Loss', avg_loss, num_iter)
                    losses["num_iter"].append(num_iter)
                    losses["loss"].append(avg_loss)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(self.tag + tag, value.data.cpu().numpy(), num_iter)
                        writer.add_histogram(self.tag + tag + '/grad', value.grad.data.cpu().numpy(), num_iter)
                    if self.evaluate_val_each_step:
                        val_loss = self.validate()
                        print("Validation: epoch %d, iter %d, total_loss: %.2f, time: %.2f"
                              % (epoch, num_iter, val_loss, time.time() - start_time))
                        writer.add_scalar(self.tag + '/ValEachStep/Loss', val_loss, num_iter)
                        val_losses["num_iter"].append(num_iter)
                        val_losses["loss"].append(val_loss)
                        self.dataloader.set_data_set("train")
                        model.train()
                    torch.save(model.state_dict(), "parameter/" + self.tag + str(num_iter))
        self.save_single_losses(losses, "Train")
        if self.evaluate_val_each_step:
            self.save_losses(losses, val_losses, "ValEachStep")
        return avg_loss, val_loss

    def load_model(self, num_iter=None):
        num_iter_str = "" if num_iter is None else str(num_iter)
        self.model.load_state_dict(torch.load("parameter/" + self.tag + num_iter_str))
        self.num_iter = num_iter
        return self

    def validate(self):
        self.dataloader.set_data_set("validation")
        losses = {"num_iter": [], "loss": []}
        writer = self.writer
        total_loss = 0
        avg_loss = 0
        model = self.model
        model.eval()
        loss_function = self.loss_function
        num_iter = 0
        start_time = time.time()

        batch_generator = self.dataloader.generate_batches(self.batch_size, shuffle=False)
        for batch_index, batch_molecule in enumerate(batch_generator):
            num_iter += 1
            features = self.dataloader.to_tensor(batch_molecule['feature'])
            response = self.dataloader.to_tensor(batch_molecule['response'])
            length = batch_molecule['length']
            y_pred = model(features, length)
            loss = loss_function(y_pred, response, length)
            loss_batch = loss.item()
            total_loss += loss_batch

            log_every = math.ceil(self.log_every / 10)
            avg_loss = total_loss / num_iter
            if not self.evaluate_val_each_step and num_iter % log_every == 0:
                print("Validation: iter %d, total_loss: %.2f, time: %.2f"
                      % (num_iter, avg_loss, time.time() - start_time))
                # print("y_pred: " + str(y_pred))
                # print("response: " + str(response))
                writer.add_histogram(self.tag + "/Val/prediction", y_pred, num_iter)
                writer.add_histogram(self.tag + "/Val/true", response, num_iter)
                writer.add_scalar(self.tag + '/Val/Loss', avg_loss, num_iter)
                writer.add_scalar(self.tag + '/Val/Loss', avg_loss, num_iter)
                losses["num_iter"].append(num_iter)
                losses["loss"].append(avg_loss)
        if not self.evaluate_val_each_step:
            self.save_single_losses(losses, "Val")
        return avg_loss

    def test(self):
        self.dataloader.set_data_set("test")
        model = self.model
        model.eval()
        results = {"id": [], "scalar_coupling_constant": []}

        batch_generator = self.dataloader.generate_batches(self.batch_size, shuffle=False, drop_last=False)
        for batch_index, batch_molecule in enumerate(batch_generator):
            features = self.dataloader.to_tensor(batch_molecule['feature'])
            length = batch_molecule['length']
            list_ids = batch_molecule['id']
            y_pred = model(features, length)
            y_pred = torch.Tensor.numpy(y_pred.cpu().detach())
            for ids, preds in zip(list_ids, y_pred):
                for _id, pred in zip(ids, preds):
                    if _id != -1:
                        results["id"].append(_id)
                        results["scalar_coupling_constant"].append(pred)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv("result/" + self.tag + str(self.num_iter) + ".csv", index=False)
        return results

    def save_single_losses(self, losses, name):
        loss_df = pd.DataFrame.from_dict(losses)
        filename = "loss/" + self.tag + name
        loss_df.to_csv(filename + ".csv", index=False)
        fig = plt.figure(figsize=(6, 4), tight_layout=True)
        ax = plt.gca()
        loss_df.plot(kind='line', x='num_iter', y='loss', ax=ax)
        fig.savefig(filename + '.png', dpi=fig.dpi)

    def save_losses(self, losses, val_losses, name):
        loss_df = pd.DataFrame.from_dict(losses)
        val_losses_df = pd.DataFrame.from_dict(val_losses)
        filename = "loss/" + self.tag + name
        val_losses_df.to_csv(filename + ".csv", index=False)
        fig = plt.figure(figsize=(6, 4), tight_layout=True)
        ax = plt.gca()
        loss_df.plot(kind='line', x='num_iter', y='loss', ax=ax, label="Train")
        val_losses_df.plot(kind='line', x='num_iter', y='loss', ax=ax, label="Validation")
        fig.savefig(filename + '.png', dpi=fig.dpi)

    def set_val_each_step(self, on):
        self.evaluate_val_each_step = on
