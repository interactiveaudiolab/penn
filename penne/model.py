import functools

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import penne
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import math

###############################################################################
# Model definition
###############################################################################

class Model(pl.LightningModule):
    """PyTorch Lightning model definition"""

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
        if penne.LOSS_FUNCTION == 'BCE' or penne.ORIGINAL_CREPE:
            if penne.SMOOTH_TARGETS or penne.ORIGINAL_CREPE:
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
        if not self.trainer.running_sanity_check:
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
                checkpoint_path = penne.CHECKPOINT_DIR.joinpath('crepe', self.name, str(self.current_epoch)+'.ckpt')
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
                # probabilities = torch.nn.Softmax(dim=1)(logits)
                self.write_posterior_distribution(logits)  

        self.val_rmse.reset()
        self.val_rpa.reset()
        self.val_rca.reset()

    def ex_batch_for_logging(self, dataset):
        if dataset == 'PTDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/PTDB/FEMALE/MIC/F01/mic_F01_sa1.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))
        elif dataset == 'MDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/MDB/audio_stems/MusicDelta_InTheHalloftheMountainKing_STEM_03.RESYN.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            # limit length to avoid memory error
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))[:1200,:]

    def write_posterior_distribution(self, probabilities):
        # plot the posterior distribution for ex_batch
        checkpoint_label = str(self.current_epoch)+'.ckpt'
        fig = plt.figure(figsize=(12, 3))
        plt.imshow(probabilities.detach().numpy().T, origin='lower')
        plt.title(checkpoint_label)
        self.logger.experiment.add_figure('output distribution', fig, global_step=self.current_epoch)

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
            self.logger.experiment.add_figure('logits', fig, global_step=self.current_epoch)


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



class NVDModel(pl.LightningModule):
    """PyTorch Lightning model definition"""

    def __init__(self, name='default'):
        super().__init__()

        self.epsilon = 0.0010000000474974513
        self.learning_rate = 2e-4

        # equivalent to Keras default momentum
        self.momentum = 0.01

        self.name = name
        self.ex_logits = None
        self.ex_targets = None
        self.ex_unvoiced = None
        self.best_loss = float('inf')

        # metrics
        self.train_rmse = penne.metrics.WRMSE()
        self.train_rpa = penne.metrics.RPA()
        self.train_rca = penne.metrics.RCA()
        self.val_rmse = penne.metrics.WRMSE()
        self.val_rpa = penne.metrics.RPA()
        self.val_rca = penne.metrics.RCA()

        in_channels = [460, 256, 256, 256, 256]
        out_channels = [256, 256, 256, 256, 360]

        # Shared layer parameters
        # kernel_sizes = [1] * 4 + [1]
        kernel_sizes = [11] * 4 + [1]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm1d,
                                          eps=self.epsilon,
                                          momentum=self.momentum)

        # Layer definitions
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2)
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv1d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv1d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2)
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv1d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            padding=kernel_sizes[3] // 2)
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv1d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            padding=kernel_sizes[4] // 2)


    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self, x, ar=None):
        """Perform model inference"""
        # Forward pass through first five layers
        # (batch, 100, 1)
        ar = ar.permute(0, 2, 1)
        # ar_voice = ar_voice.permute(0, 2, 1)
        # (batch, 460, 1)
        x = torch.cat((x, ar), axis=1)
        
        x = self.layer(x, self.conv1, self.conv1_BN)
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        return self.conv5(x)

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
        # y_hat, y [batch, 2, time]
        # 0 -> pitch
        # mse_loss = F.mse_loss(y_hat[:,0], y[:,0]) #log2 hz
        # # mse_loss = F.mse_loss(y_hat[:,0][y[:,1]==1], y[:,0][y[:,1]==1])
        # # bce_loss = F.binary_cross_entropy_with_logits(y_hat[:,1], y[:,1]) #periodicity stuff (optional)

        # # maybe weight these
        # return mse_loss #+ bce_loss
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, index):
        """Performs one step of training"""
        x, y, ar = batch
        output = self(x, ar[:,0:1])
        loss = self.my_loss(output, y[:,0])

        # update epoch's cumulative rmse, rpa, rca with current batch
        np_y_hat = output.argmax(axis=1).cpu().detach().numpy()
        np_y = y[:,0].cpu().numpy()
        np_voicing = y[:,1].cpu().numpy()
        self.train_rpa.update(np_y_hat, np_y, voicing=np_voicing)
        self.train_rca.update(np_y_hat, np_y, voicing=np_voicing)
        return {"loss": loss}

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        x, y, ar = batch
        output = self(x, ar[:,0:1])
        loss = self.my_loss(output, y[:,0])

        # update epoch's cumulative rmse, rpa, rca with current batch
        np_y_hat = output.argmax(axis=1).cpu().detach().numpy()
        np_y = y[:,0].cpu().numpy()
        np_voicing = y[:,1].cpu().numpy()
        self.val_rpa.update(np_y_hat, np_y, voicing=np_voicing)
        self.val_rca.update(np_y_hat, np_y, voicing=np_voicing)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # compute mean loss and accuracy
        loss_sum = 0
        for x in outputs:
            loss_sum += x['loss']
        loss_mean = loss_sum / len(outputs)

        # log metrics to tensorboard
        self.logger.experiment.add_scalar("Loss/Train", loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar("RPA/Train", self.train_rpa(), self.current_epoch)
        self.logger.experiment.add_scalar("RCA/Train", self.train_rca(), self.current_epoch)

        # reset metrics for next epoch
        self.train_rpa.reset()
        self.train_rca.reset()

    def validation_epoch_end(self, outputs):
        # compute mean loss and accuracy
        if not self.trainer.running_sanity_check:
            loss_sum = 0
            for x in outputs:
                loss_sum += x['loss']
            loss_mean = loss_sum / len(outputs)

            # log metrics to tensorboard
            self.logger.experiment.add_scalar("Loss/Val", loss_mean, self.current_epoch)
            self.logger.experiment.add_scalar("RPA/Val", self.val_rpa(), self.current_epoch)
            self.logger.experiment.add_scalar("RCA/Val", self.val_rca(), self.current_epoch)

            # log mean validation loss for early stopping
            self.log('val_loss', loss_mean)

            # save the best checkpoint so far
            if (loss_mean < self.best_loss or self.current_epoch % 100 == 0) and self.current_epoch > 5:
                self.best_loss = loss_mean
                checkpoint_path = penne.CHECKPOINT_DIR.joinpath('nvd', self.name, str(self.current_epoch)+'.ckpt')
                self.trainer.save_checkpoint(checkpoint_path)

            if self.current_epoch < 20 or self.current_epoch % 25 == 0:
            # load a batch for logging if not yet loaded
                if self.ex_logits is None:
                    self.ex_logits = torch.load(penne.data.stem_to_nvd_logits('PTDB', 'F01_sa1'), 'cuda')[None, :]
                    self.ex_targets = torch.load(penne.data.stem_to_nvd_targets('PTDB', 'F01_sa1'))

                    # for argmax
                    # self.ex_targets[0] = self.ex_logits.argmax(axis=1)

                    self.ex_unvoiced = self.ex_targets[1].cpu().numpy().squeeze()==0
                    self.ex_targets = penne.convert.frequency_to_bins(2**self.ex_targets).float()
                    # self.ex_voicing = self.ex_targets[1].cpu().numpy().squeeze()
                    self.ex_targets = self.ex_targets[0].cpu().numpy().squeeze()
                    self.ex_targets[self.ex_unvoiced] = np.nan

                pitches, pitch_dist, entropies = penne.ar_loop(self, self.ex_logits)

                # preds = self(self.ex_logits)
                preds = pitches.cpu().detach().numpy().squeeze()
                preds[self.ex_unvoiced] = np.nan

                checkpoint_label = str(self.current_epoch)+'.ckpt'
                fig = plt.figure(figsize=(6, 3))
                plt.plot(preds, label='predictions')
                plt.plot(self.ex_targets, label='targets')
                plt.legend()
                plt.title(checkpoint_label)
                self.logger.experiment.add_figure('predictions', fig, global_step=self.current_epoch)

                dist_fig = plt.figure(figsize=(6, 3))
                plt.imshow(pitch_dist.cpu().squeeze())
                self.logger.experiment.add_figure("output distribution", dist_fig, global_step=self.current_epoch)


        self.val_rpa.reset()
        self.val_rca.reset()


    ###########################################################################
    # PyTorch Lightning - optimizer
    ###########################################################################

    def configure_optimizers(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def layer(self, x, conv, batch_norm):
        """Forward pass through one layer"""
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.dropout(x, p=0.25, training=self.training)


class ARModel(pl.LightningModule):
    """PyTorch Lightning model definition"""

    def __init__(self, name='default'):
        super().__init__()

        self.epsilon = 0.0010000000474974513
        self.learning_rate = 2e-4

        # equivalent to Keras default momentum
        self.momentum = 0.01

        self.name = name
        self.ex_frames = None
        self.best_loss = float('inf')

        # metrics
        thresh = penne.threshold.Hysteresis()
        self.train_rmse = penne.metrics.WRMSE()
        self.train_rpa = penne.metrics.RPA()
        self.train_rca = penne.metrics.RCA()
        self.val_rmse = penne.metrics.WRMSE()
        self.val_rpa = penne.metrics.RPA()
        self.val_rca = penne.metrics.RCA()
        
        # subnet
        # ar (batch, 1, 100) -> (batch, k)

        # first in channel k+1

        # input (batch, k+1, 1024)

        # in_channels = [1, 1024, 128, 128, 128, 256] 
        in_channels = [1+penne.AR_OUTPUT_SIZE, 1024, 128, 128, 128, 256]
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

        self.ar_model = Autoregressive()

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self, x, ar, embed=False):
        """Perform model inference"""
        # x: (batch, window_size) -> (batch, 1, window_size)
        x = x[:, None, :]

        # ar -> (batch, ar_size), ar_feats -> (batch, ar_output_size)
        ar_feats = self.ar_model(ar)
        # ar_feats -> (batch, ar_output_size, window_size)
        ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2])

        # x -> (batch, 1+ar_output_size, window_size, 1)
        x = torch.cat((x, ar_feats), dim=1).unsqueeze(-1)

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
        # apply Gaussian blur around target bin
        mean = penne.convert.bins_to_cents(y)
        normal = torch.distributions.Normal(mean, 25)
        bins = penne.convert.bins_to_cents(torch.arange(penne.PITCH_BINS).to(y.device))
        bins = bins[:, None]
        y = torch.exp(normal.log_prob(bins)).permute(1,0)
        y /= y.max(dim=1, keepdims=True).values
        assert y_hat.shape == y.shape
        return F.binary_cross_entropy_with_logits(y_hat, y.float())
        # return F.cross_entropy(y_hat, y.long())

    def my_acc(self, y_hat, y):
        argmax_y_hat = y_hat.argmax(dim=1)
        return argmax_y_hat.eq(y).sum().item()/y.numel()

    def training_step(self, batch, index):
        """Performs one step of training"""
        x, y, ar = batch
        output = self(x, ar)
        loss = self.my_loss(output, y)
        acc = self.my_acc(output, y)

        # update epoch's cumulative rmse, rpa, rca with current batch
        y_hat = output.argmax(dim=1)
        np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
        np_y = y.cpu().numpy()[None,:]
        self.train_rpa.update(np_y_hat_freq, np_y)
        self.train_rca.update(np_y_hat_freq, np_y)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        x, y, ar = batch
        output = self(x, ar)
        loss = self.my_loss(output, y)
        acc = self.my_acc(output, y)

        # update epoch's cumulative rmse, rpa, rca with current batch
        y_hat = output.argmax(dim=1)
        np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
        np_y = y.cpu().numpy()[None,:]
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
        self.logger.experiment.add_scalar("RPA/Train", self.train_rpa(), self.current_epoch)
        self.logger.experiment.add_scalar("RCA/Train", self.train_rca(), self.current_epoch)

        # reset metrics for next epoch
        self.train_rpa.reset()
        self.train_rca.reset()

    def validation_epoch_end(self, outputs):
        # compute mean loss and accuracy
        if not self.trainer.running_sanity_check:
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
            self.logger.experiment.add_scalar("RPA/Val", self.val_rpa(), self.current_epoch)
            self.logger.experiment.add_scalar("RCA/Val", self.val_rca(), self.current_epoch)

            # log mean validation accuracy for early stopping
            self.log('val_accuracy', acc_mean)

            # save the best checkpoint so far
            if loss_mean < self.best_loss and self.current_epoch > 5:
                self.best_loss = loss_mean
                checkpoint_path = penne.CHECKPOINT_DIR.joinpath('ar', self.name, str(self.current_epoch)+'.ckpt')
                self.trainer.save_checkpoint(checkpoint_path)

        if self.current_epoch % 20 == 0:
        # load a batch for logging if not yet loaded
            if self.ex_frames is None:
                dataset, stem = 'PTDB', 'F01_sa1'
                # dataset, stem = 'MDB', 'MusicDelta_InTheHalloftheMountainKing_STEM_03'
                self.ex_frames = torch.from_numpy(np.load(penne.data.stem_to_cache_frames(dataset, stem, False))).to(self.device)
                self.ex_frames -= self.ex_frames.mean(dim=1, keepdim=True)
                self.ex_frames /= torch.max(torch.tensor(1e-10, device=self.ex_frames.device),
                    self.ex_frames.std(dim=1, keepdim=True))
                self.ex_targets = torch.load(penne.data.stem_to_nvd_targets(dataset, stem))

                self.ex_unvoiced = self.ex_targets[1].cpu().numpy().squeeze()==0
                self.ex_targets = penne.convert.frequency_to_bins(2**self.ex_targets).float()
                self.ex_targets = self.ex_targets[0].cpu().numpy().squeeze()
                self.ex_targets[self.ex_unvoiced] = np.nan

            pitches, pitch_dist, entropies = penne.ar_loop(self, self.ex_frames)

            preds = pitches.cpu().detach().numpy().squeeze()
            preds[self.ex_unvoiced] = np.nan

            checkpoint_label = str(self.current_epoch)+'.ckpt'
            fig = plt.figure(figsize=(6, 3))
            plt.plot(preds, label='predictions')
            plt.plot(self.ex_targets, label='targets')
            plt.legend()
            plt.title(checkpoint_label)
            self.logger.experiment.add_figure('predictions', fig, global_step=self.current_epoch)

            dist_fig = plt.figure(figsize=(6, 3))
            plt.imshow(pitch_dist.cpu().squeeze())
            plt.gca().invert_yaxis()
            self.logger.experiment.add_figure("output distribution", dist_fig, global_step=self.current_epoch)

        self.val_rpa.reset()
        self.val_rca.reset()


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
        # x.shape=(batch, 1+ar_output_size, 1024, 1)
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

class Autoregressive(torch.nn.Module):
    def __init__(self):
        super().__init__()

        conv_sizes = [128, 64, 16, 4]
        kernel_size = 5
        self.linear_input_size = 400

        self.embedding = torch.nn.Embedding(penne.PITCH_BINS+2, conv_sizes[0])

        conv_layers = [
            torch.nn.Conv1d(conv_sizes[0], conv_sizes[1], kernel_size, padding=kernel_size // 2),
            torch.nn.Conv1d(conv_sizes[1], conv_sizes[2], kernel_size, padding=kernel_size // 2),
            torch.nn.Conv1d(conv_sizes[2], conv_sizes[3], kernel_size, padding=kernel_size // 2),
            # torch.nn.Linear(penne.AR_SIZE, penne.AR_HIDDEN_SIZE),
            torch.nn.LeakyReLU(.1)]
        self.conv_layers = torch.nn.Sequential(*conv_layers)
            
        
        linear_layers = [
            torch.nn.Linear(self.linear_input_size, penne.AR_HIDDEN_SIZE),
            torch.nn.LeakyReLU(.1)]
        for _ in range(2):
            linear_layers.extend([
                torch.nn.Linear(
                    penne.AR_HIDDEN_SIZE,
                    penne.AR_HIDDEN_SIZE),
                torch.nn.LeakyReLU(.1)])
        linear_layers.append(
            torch.nn.Linear(penne.AR_HIDDEN_SIZE, penne.AR_OUTPUT_SIZE))
        self.linear_layers = torch.nn.Sequential(*linear_layers)
    
    def forward(self, x):
        # x = torch.full(x.shape, 0, dtype=x.dtype, device=x.device)
        # x.shape = (batch, 100)
        x = self.embedding(x) # (batch, 100, 128)
        x = x.permute(0, 2, 1) # (batch, 128, 100)
        x = self.conv_layers(x) # (batch, 4, 100)
        x = x.reshape(-1, self.linear_input_size) # (batch, 400)
        return self.linear_layers(x) # (batch, 64)

class PDCModel(pl.LightningModule):
    """PyTorch Lightning model definition"""

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
        # kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        kernel_sizes = [(15, 1)] + 5 * [(15, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]
        
        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=self.epsilon,
                                          momentum=self.momentum)

        # Layer definitions
        self.conv1 = PrimeDilatedConvolutionBlock(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        # self.conv1 = torch.nn.Linear(
        #     in_features=in_channels[0],
        #     out_features=out_channels[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = PrimeDilatedConvolutionBlock(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = PrimeDilatedConvolutionBlock(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = PrimeDilatedConvolutionBlock(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = PrimeDilatedConvolutionBlock(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = PrimeDilatedConvolutionBlock(
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
        if penne.LOSS_FUNCTION == 'BCE' or penne.ORIGINAL_CREPE:
            if penne.SMOOTH_TARGETS or penne.ORIGINAL_CREPE:
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
        if not self.trainer.running_sanity_check:
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
                checkpoint_path = penne.CHECKPOINT_DIR.joinpath('pdc', self.name, str(self.current_epoch)+'.ckpt')
                self.trainer.save_checkpoint(checkpoint_path)

            # plot logits and posterior distribution
            if self.current_epoch < 20 or self.current_epoch % 25 == 0:
                # load a batch for logging if not yet loaded
                if self.ex_batch is None:
                    self.ex_batch = self.ex_batch_for_logging('PTDB')

                # plot logits
                logits = penne.infer(self.ex_batch, model=self).cpu()

                # plot posterior distribution
                # probabilities = torch.nn.Softmax(dim=1)(logits)
                self.write_posterior_distribution(logits)  

        self.val_rmse.reset()
        self.val_rpa.reset()
        self.val_rca.reset()

    def ex_batch_for_logging(self, dataset):
        if dataset == 'PTDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/PTDB/FEMALE/MIC/F01/mic_F01_sa1.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))
        elif dataset == 'MDB':
            ex_audio, ex_sr = penne.load.audio("/home/caedon/penne/data/MDB/audio_stems/MusicDelta_InTheHalloftheMountainKing_STEM_03.RESYN.wav")
            ex_audio = penne.resample(ex_audio, ex_sr)
            # limit length to avoid memory error
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device='cuda'))[:1200,:]

    def write_posterior_distribution(self, probabilities):
        # plot the posterior distribution for ex_batch
        checkpoint_label = str(self.current_epoch)+'.ckpt'
        fig = plt.figure(figsize=(12, 3))
        plt.imshow(probabilities.detach().numpy().T, origin='lower')
        plt.title(checkpoint_label)
        self.logger.experiment.add_figure('output distribution', fig, global_step=self.current_epoch)

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
        x = self.layer(x, self.conv1, self.conv1_BN)
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)
        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 0, 0), pooling=(2, 1), linear=False):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        if linear:
            x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = batch_norm(x)
        x = F.max_pool2d(x, pooling, pooling)
        return F.dropout(x, p=0.25, training=self.training)


# List of prime numbers for dilations
# And yes, 1 is technically not a prime number
PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


###############################################################################
# Constants
###############################################################################


class PrimeDilatedConvolutionBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        primes=4):
        super().__init__()

        # Require output_channels to be evenly divisble by number of primes
        if out_channels // primes * primes != out_channels:
            raise ValueError(
                'output_channels must be evenly divisble by number of primes')

        # Initialize layers
        if stride == (1, 1):
            conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        else:
            conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            stride=stride)
        self.layers = torch.nn.ModuleList([
            conv_fn(in_channels, out_channels // primes, dilation=prime)
            for prime in PRIMES[:primes]])
        
        self.stride = stride

    def forward(self, x):
        """Forward pass"""
        # Forward pass through each
        if self.stride == (1, 1):
            return torch.cat(tuple(layer(x) for layer in self.layers), dim=1)
        else:
            layer_outs = []
            for i, layer in enumerate(self.layers):
                dilation = PRIMES[i]
                L_in = x.shape[2]
                L_out = L_in // self.stride[0]
                padding = math.ceil((self.stride[0] * (L_out - 1) - L_in + dilation * (layer.kernel_size[0] - 1) + 1) / 2)
                layer_out = layer(F.pad(x, (0, 0, padding, padding)))
                layer_outs.append(layer_out)
            return torch.cat(layer_outs, dim=1)