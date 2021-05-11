import functools

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import penne
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Model definition
###############################################################################

class Model(pl.LightningModule):
    """PyTorch Lightning model definition"""

    # TODO - add hyperparameters as input args
    def __init__(self, name='default'):
        super().__init__()

        self.epsilon = 0.0010000000474974513
        self.learning_rate = 2e-4

        # equivalent to Keras default momentum
        self.momentum = 0.01

        self.name = name
        self.ex_batch = None
        self.best_loss = float('inf')

        # metrics
        thresh = penne.threshold.Hysteresis()
        self.train_rmse = penne.metrics.WRMSE()
        self.train_rpa = penne.metrics.RPA()
        self.train_rca = penne.metrics.RCA()
        self.val_rmse = penne.metrics.WRMSE()
        self.val_rpa = penne.metrics.RPA()
        self.val_rca = penne.metrics.RCA()

        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        self.in_features = 2048

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=self.epsilon,
                                          momentum=self.momentum)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=penne.PITCH_BINS)

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self, x, embed=False):
        """Perform model inference"""
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return self.classifier(x)

    ###########################################################################
    # PyTorch Lightning - model-specific argparse argument hook
    ###########################################################################

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model hyperparameters as argparse arguments"""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    ###########################################################################
    # PyTorch Lightning - step model hooks
    ###########################################################################

    def my_loss(self, y_hat, y):
        if penne.LOSS_FUNCTION == 'BCE':
            if penne.SMOOTH_TARGETS:
                # apply Gaussian blur around target bin
                mean = penne.convert.bins_to_cents(y)
                normal = torch.distributions.Normal(mean, 25)
                bins = penne.convert.bins_to_cents(torch.arange(penne.PITCH_BINS).to(y.device))
                bins = bins[:, None]
                y = torch.exp(normal.log_prob(bins)).permute(1,0)
                y /= y.max(dim=1, keepdims=True).values
            else:
                y = F.one_hot(y, penne.PITCH_BINS)
            assert y_hat.shape == y.shape
            return F.binary_cross_entropy_with_logits(y_hat, y.float())
        return F.cross_entropy(y_hat, y.long())

    def my_acc(self, y_hat, y):
        argmax_y_hat = y_hat.argmax(dim=1)
        return argmax_y_hat.eq(y).sum().item()/y.numel()

    def topk_acc(self, y_hat, y, k):
        total = 0
        for i in range(k):
            total += torch.topk(y_hat, k, dim=1)[1][:,i].eq(y).sum().item()
        return total / y.numel()

    def training_step(self, batch, index):
        """Performs one step of training"""
        x, y = batch
        output = self(x)
        loss = self.my_loss(output, y)
        acc = self.my_acc(output, y)

        # update epoch's cumulative rmse, rpa, rca with current batch
        y_hat = output.argmax(dim=1)
        np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
        np_y = y.cpu().numpy()[None,:]
        self.train_rmse.update(np_y_hat_freq, np_y, np.ones(np_y.shape))
        self.train_rpa.update(np_y_hat_freq, np_y)
        self.train_rca.update(np_y_hat_freq, np_y)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        x, y = batch
        output = self(x)
        loss = self.my_loss(output, y)
        acc = self.my_acc(output, y)

        # update epoch's cumulative rmse, rpa, rca with current batch
        y_hat = output.argmax(dim=1)
        np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
        np_y = y.cpu().numpy()[None,:]
        self.val_rmse.update(np_y_hat_freq, np_y, np.ones(np_y.shape))
        self.val_rpa.update(np_y_hat_freq, np_y)
        self.val_rca.update(np_y_hat_freq, np_y)
        return {"loss": loss, "accuracy": acc}

    def training_epoch_end(self, outputs):
        # compute mean loss and accuracy
        loss_sum = 0
        acc_sum = 0
        for x in outputs:
            loss_sum += x['loss']
            acc_sum += x['accuracy']
        loss_mean = loss_sum / len(outputs)
        acc_mean = acc_sum / len(outputs)

        # log metrics to tensorboard
        self.logger.experiment.add_scalar("Loss/Train", loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", acc_mean, self.current_epoch)
        self.logger.experiment.add_scalar("RMSE/Train", self.train_rmse(), self.current_epoch)
        self.logger.experiment.add_scalar("RPA/Train", self.train_rpa(), self.current_epoch)
        self.logger.experiment.add_scalar("RCA/Train", self.train_rca(), self.current_epoch)

        # reset metrics for next epoch
        self.train_rmse.reset()
        self.train_rpa.reset()
        self.train_rca.reset()

    def validation_epoch_end(self, outputs):
        # compute mean loss and accuracy
        loss_sum = 0
        acc_sum = 0
        for x in outputs:
            loss_sum += x['loss']
            acc_sum += x['accuracy']
        loss_mean = loss_sum / len(outputs)
        acc_mean = acc_sum / len(outputs)

        # log metrics to tensorboard
        self.logger.experiment.add_scalar("Loss/Val", loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val", acc_mean, self.current_epoch)
        self.logger.experiment.add_scalar("RMSE/Val", self.val_rmse(), self.current_epoch)
        self.logger.experiment.add_scalar("RPA/Val", self.val_rpa(), self.current_epoch)
        self.logger.experiment.add_scalar("RCA/Val", self.val_rca(), self.current_epoch)

        # log mean validation accuracy for early stopping
        self.log('val_accuracy', acc_mean)

        # save the best checkpoint so far
        if loss_mean < self.best_loss and self.current_epoch > 5:
            self.best_loss = loss_mean
            checkpoint_path = penne.CHECKPOINT_DIR.joinpath(self.name, str(self.current_epoch)+'.ckpt')
            self.trainer.save_checkpoint(checkpoint_path)

        # plot logits and posterior distribution
        if self.current_epoch < 20 or self.current_epoch % 25 == 0:
            # load a batch for logging if not yet loaded
            if self.ex_batch is None:
                self.ex_batch = self.ex_batch_for_logging('PTDB')

            # plot logits
            logits = penne.infer(self.ex_batch, model=self).cpu()
            self.write_logits(logits, [10, 80, 400, 650])

            # plot posterior distribution
            probabilities = torch.nn.Softmax(dim=1)(logits)
            self.write_posterior_distribution(probabilities)  

        self.val_rmse.reset()
        self.val_rpa.reset()
        self.val_rca.reset()

    def ex_batch_for_logging(self, dataset):
        if dataset == 'PTDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/PTDB/MALE/MIC/M03/mic_M03_sa1.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            return next(penne.preprocess(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))
        elif dataset == 'MDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/MDB/audio_stems/MusicDelta_InTheHalloftheMountainKing_STEM_03.RESYN.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            # limit length to avoid memory error
            return next(penne.preprocess(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))[:1200,:]

    def write_posterior_distribution(self, probabilities):
        # plot the posterior distribution for ex_batch
        checkpoint_label = str(self.current_epoch)+'.ckpt'
        fig = plt.figure(figsize=(12, 3))
        plt.imshow(probabilities.detach().numpy().T, origin='lower')
        plt.title(checkpoint_label)
        self.logger.experiment.add_figure(checkpoint_label, fig, global_step=self.current_epoch)

    def write_logits(self, logits, frames):
        # plot the logits for the 4 indexed frames in frames
        if len(frames) == 4:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
            ax1.plot(logits[frames[0]].detach().numpy())
            ax1.set_title('Frame ' + str(frames[0]))
            ax2.plot(logits[frames[1]].detach().numpy())
            ax2.set_title('Frame ' + str(frames[1]))
            ax3.plot(logits[frames[2]].detach().numpy())
            ax3.set_title('Frame ' + str(frames[2]))
            ax4.plot(logits[frames[3]].detach().numpy())
            ax4.set_title('Frame ' + str(frames[3]))
            checkpoint_label = str(self.current_epoch)+'.ckpt'
            self.logger.experiment.add_figure(checkpoint_label+' logits', fig, global_step=self.current_epoch)


    ###########################################################################
    # PyTorch Lightning - optimizer
    ###########################################################################

    def configure_optimizers(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]
        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)
        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))
        return F.dropout(x, p=0.25, training=self.training)
