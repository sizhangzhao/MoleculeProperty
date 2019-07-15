from abc import abstractmethod

import torch
from torch.nn.modules.loss import L1Loss
import time
import pandas as pd
from tensorboardX import SummaryWriter
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from enum import Enum


class DataSet(Enum):
    TRAIN = "train",
    VAL = "validation",
    TEST = "test"


class BaseTrainer:

    def __init__(self, num_epoch, batch_size, log_every=100, lr=0.01, dropout_ratio=0.5,
                 clip_grad=0.5, evaluate_val_each_step=False):
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.clip_grad = clip_grad
        self.batch_size = batch_size
        self.log_every = log_every
        self.dropout_ratio = dropout_ratio
        self.dataloader, self.model, self.loss_function = self.create_model()
        self.move_device()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.writer = SummaryWriter('log')
        self.evaluate_val_each_step = evaluate_val_each_step
        self.generate_basic_model_information()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def update_tag(self):
        pass

    @abstractmethod
    def get_response(self, batch):
        pass

    @abstractmethod
    def get_id(self, batch):
        pass

    @abstractmethod
    def to_tensor(self, batch):
        pass

    def generate_basic_model_information(self):
        self.tag = "e" + str(self.num_epoch) + "c" + str(self.clip_grad) + "lr" + str(self.lr) + "b" \
                   + str(self.batch_size)
        self.num_iter = 0

    def move_device(self):
        # if torch.cuda.device_count() > 1 and self.device != "cpu":
        #     self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def change_model(self):
        self.model = self.create_model()
        self.move_device()
        self.generate_basic_model_information()
        self.update_tag()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.change_model()

    def set_lr(self, lr):
        self.lr = lr
        self.change_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        self.dataloader.set_data_set(DataSet.TRAIN)
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
                batch_molecule = self.to_tensor(batch_molecule)
                num_iter += 1
                optimizer.zero_grad()
                response = self.get_response(batch_molecule)
                y_pred = model(batch_molecule)
                loss = loss_function(y_pred, response)
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
        self.dataloader.set_data_set(DataSet.VAL)
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
            batch_molecule = self.to_tensor(batch_molecule)
            num_iter += 1
            response = self.get_response(batch_molecule)
            y_pred = model(batch_molecule)
            loss = loss_function(y_pred, response)
            loss_batch = loss.item()
            total_loss += loss_batch

            log_every = math.ceil(self.log_every / 10)
            avg_loss = total_loss / num_iter
            if not self.evaluate_val_each_step and num_iter % log_every == 0:
                print("Validation: iter %d, total_loss: %.2f, time: %.2f"
                      % (num_iter, avg_loss, time.time() - start_time))
                writer.add_histogram(self.tag + "/Val/prediction", y_pred, num_iter)
                writer.add_histogram(self.tag + "/Val/true", response, num_iter)
                writer.add_scalar(self.tag + '/Val/Loss', avg_loss, num_iter)
                writer.add_scalar(self.tag + '/Val/Loss', avg_loss, num_iter)
                losses["num_iter"].append(num_iter)
                losses["loss"].append(avg_loss)
        if not self.evaluate_val_each_step:
            self.save_single_losses(losses, "Val")
        print("Validation: total_loss: %.2f, time: %.2f" % (avg_loss, time.time() - start_time))
        return avg_loss

    def test(self):
        self.dataloader.set_data_set(DataSet.TEST)
        model = self.model
        model.eval()
        results = {"id": [], "scalar_coupling_constant": []}

        batch_generator = self.dataloader.generate_batches(self.batch_size, shuffle=False, drop_last=False)
        for batch_index, batch_molecule in enumerate(batch_generator):
            batch_molecule = self.to_tensor(batch_molecule)
            list_ids = self.get_id(batch_molecule)
            y_pred = model(batch_molecule)
            y_pred = torch.Tensor.numpy(y_pred.cpu().detach()).tolist()
            for id, pred in zip(list_ids, y_pred):
                results["id"].append(id)
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

    def to_tensor_one(self, tensor):
        return tensor.contiguous().to(self.device)
