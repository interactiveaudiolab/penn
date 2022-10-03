"""train.py - model training"""


import argparse
import os
from pathlib import Path
import matplotlib
import time

from tqdm import tqdm
import penne
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
import numpy as np

from penne import data


###############################################################################
# Train
###############################################################################

class AverageMeter(object):
    """
        Computes and stores the average and current value
        Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count

def my_loss(y_hat, y, blur=True, harmo=False):
    blur = True #Forces True for now
    if blur:
        # apply Gaussian blur around target bin
        mean = penne.convert.bins_to_cents(y)
        normal = torch.distributions.Normal(mean, 25)
        bins = penne.convert.bins_to_cents(torch.arange(penne.PITCH_BINS).to(y.device))
        bins = bins[:, None]
        y = torch.exp(normal.log_prob(bins)).permute(1,0)
        y /= y.max(dim=1, keepdims=True).values
    else:
        y = torch.nn.functional.one_hot(y.long(), num_classes=penne.PITCH_BINS)
    assert y_hat.shape == y.shape
    weight = torch.ones(y_hat.shape).cuda() * 20 if harmo else torch.ones(y_hat.shape).cuda()
    return F.binary_cross_entropy_with_logits(y_hat, y.float(), weight=weight)

def my_acc(y_hat, y, voicing):
    y_hat_voiced = y_hat[voicing == 1]
    y_voiced = y[voicing == 1]
    argmax_y_hat = y_hat_voiced.argmax(dim=1)
    if y_voiced.numel() == 0:
        return 0, 0
    return argmax_y_hat.eq(y_voiced).sum().item()/y_voiced.numel(), y_voiced.numel() #Also return count properly

def ex_batch_for_logging(dataset, device='cuda'):
        if dataset == 'PTDB':
            audio_file = penne.data.stem_to_file(dataset, 'F01_sa1')
            ex_audio, ex_sr = penne.load.audio(audio_file)
            ex_audio = penne.resample(ex_audio, ex_sr)
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device=device))
        elif dataset == 'MDB':
            audio_file = penne.data.stem_to_file(dataset, 'MusicDelta_InTheHalloftheMountainKing_STEM_03')
            ex_audio, ex_sr = penne.load.audio(audio_file)
            ex_audio = penne.resample(ex_audio, ex_sr)
            # limit length to avoid memory error
            return next(penne.preprocess_from_audio(ex_audio, penne.SAMPLE_RATE, penne.HOP_SIZE, device=device))[:1200,:]

def write_posterior_distribution(probabilities, writer, epoch):
    # plot the posterior distribution for ex_batch
    checkpoint_label = str(epoch)+'.ckpt'
    fig = plt.figure(figsize=(12, 3))
    plt.imshow(probabilities.detach().numpy().T, origin='lower')
    plt.title(checkpoint_label)
    writer.add_figure('output distribution', fig, global_step=epoch)

def main():
    """Train a model"""

    ###########################################################################
    # Setup
    ###########################################################################

    # Parse command-line arguments
    args = parse_args()

    # Setup early stopping for 32 epochs (by default according to CREPE) of no val accuracy improvement
    patience = penne.EARLY_STOP_PATIENCE

    num_samples = 100 if args.harmof0 else 1

    # Setup data
    datamodule = penne.data.DataModule(args.dataset,
                                args.batch_size,
                                args.num_workers,
                                num_samples)

    # Setup log directory and model according to --pdc flag
    if args.pdc:
        logdir = 'pdc'
        model = penne.PDCModel(name=args.name)
    elif args.harmof0:
        model = penne.HarmoF0()
    else:
        logdir = 'crepe'
        model = penne.Model(name=args.name)

    #Select device
    if torch.cuda.is_available() and args.gpus > -1:
        device = torch.device('cuda', args.gpus)
    else:
        device = torch.device('cpu')
    model = model.to(device)
    lr = penne.HARMO_LEARNING_RATE if args.harmof0 else penne.LEARNING_RATE 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    
    #If early_stop_count gets above patience, stop early
    early_stop_count = 0
    last_val_acc = 0

    model.train() #Model in training mode

    

    train_rmse = penne.metrics.WRMSE()
    train_rpa = penne.metrics.RPA()
    train_rca = penne.metrics.RCA()
    val_rmse = penne.metrics.WRMSE()
    val_rpa = penne.metrics.RPA()
    val_rca = penne.metrics.RCA()

    writer = SummaryWriter(penne.RUNS_DIR / "logs" / args.name)

    ex_batch = None

    #Checkpointing setup

    best_loss = float('inf')

    cp_path = 'pdc' if args.pdc else ('harmof0' if args.harmof0 else 'crepe')
    checkpoint_dir = penne.CHECKPOINT_DIR / cp_path / args.name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    if not args.restart:
        checkpoint_file = checkpoint_dir / 'latest.ckpt'
        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['val_loss']
            print("Resuming training from epoch " + str(epoch))
        else:
            epoch = 1
    else:
        epoch = 1
        for checkpoint in checkpoint_dir.iterdir(): checkpoint.unlink() #Clear checkpoints from previous run if restarting
    best_checkpoint_file = checkpoint_dir / 'best.ckpt'
    if best_checkpoint_file.exists():
        best_checkpoint = torch.load(checkpoint_file, map_location='cpu')
        best_loss = best_checkpoint['val_loss']
        

    ###########################################################################
    # Train loop
    ###########################################################################

    while epoch < args.max_epochs + 1:
        train_losses = AverageMeter('Loss', ':.4e')
        train_accs = AverageMeter('Accuracy', ':6.4f')
        valid_losses = AverageMeter('Loss', ':.4e')
        valid_accs = AverageMeter('Accuracy', ':6.4f')

        ###########################################################################
        # Training on each batch (from previous train_step)
        ###########################################################################

        for t, (x, y, voicing) in enumerate(tqdm(train_loader, desc='Epoch ' + str(epoch) + ' training', total=min(len(train_loader), args.limit_train_batches))):
            if t > args.limit_train_batches:
                break
            x, y, voicing = x.to(device), y.to(device), voicing.to(device)
            output = model(x)

            #If we have multiple samples (i.e. for HarmoF0), squish the batch and num_samples dimensions for loss evaluation
            if len(output.shape) == 3:
                output = output.view(-1, output.shape[-1])
                y = y.view(-1)
                voicing = voicing.view(-1)
            loss = my_loss(output, y, harmo=args.harmof0)
            acc, num_voiced = my_acc(output, y, voicing)

            # update epoch's cumulative rmse, rpa, rca with current batch
            y_hat = output.argmax(dim=1)
            np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
            np_y = y.cpu().numpy()[None,:]
            np_voicing = voicing.cpu().numpy()[None,:]
            # np_voicing masks out unvoiced frames
            train_rmse.update(np_y_hat_freq, np_y, np_voicing)
            train_rpa.update(np_y_hat_freq, np_y, voicing=np_voicing)
            train_rca.update(np_y_hat_freq, np_y, voicing=np_voicing)

            train_losses.update(loss.item(), x.size(0))
            train_accs.update(acc, num_voiced)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('training loss: %.5f, training accuracy: %.5f' % (train_losses.avg, train_accs.avg))
        
        ###########################################################################
        # Log training metrics to Tensorboard
        ###########################################################################

        writer.add_scalar("Loss/Train", train_losses.avg, epoch)
        writer.add_scalar("Accuracy/Train", train_accs.avg, epoch)
        writer.add_scalar("RMSE/Train", train_rmse(), epoch)
        writer.add_scalar("RPA/Train", train_rpa(), epoch)
        writer.add_scalar("RCA/Train", train_rca(), epoch)
        
        train_rmse.reset()
        train_rpa.reset()
        train_rca.reset()

        ###########################################################################
        # Validate on each batch (from previous validation_step)
        ###########################################################################

        with torch.no_grad():
            for t, (x, y, voicing) in enumerate(tqdm(valid_loader, desc='Epoch ' + str(epoch) + ' validation', total=min(len(valid_loader), args.limit_val_batches))):
                """Performs one step of validation"""
                if t > args.limit_val_batches:
                    break
                x, y, voicing = x.to(device), y.to(device), voicing.to(device)
                output = model(x)
                #If we have multiple samples (i.e. for HarmoF0), squish the batch and num_samples dimensions for loss evaluation
                if len(output.shape) == 3:
                    output = output.view(-1, output.shape[-1])
                    y = y.view(-1)
                    voicing = voicing.view(-1)
                loss = my_loss(output, y, harmo=args.harmof0)
                acc, num_voiced = my_acc(output, y, voicing)
                
                # update epoch's cumulative rmse, rpa, rca with current batch
                y_hat = output.argmax(dim=1)
                np_y_hat_freq = penne.convert.bins_to_frequency(y_hat).cpu().numpy()[None,:]
                np_y = y.cpu().numpy()[None,:]
                np_voicing = voicing.cpu().numpy()[None,:]
                # np_voicing masks out unvoiced frames
                val_rmse.update(np_y_hat_freq, np_y, np_voicing)
                val_rpa.update(np_y_hat_freq, np_y)
                val_rca.update(np_y_hat_freq, np_y)

                valid_losses.update(loss.item(), x.size(0))
                valid_accs.update(acc, num_voiced)
        
        print('validation loss: %.5f, validation accuracy: %.5f' % (valid_losses.avg, valid_accs.avg))
        val_accuracy = valid_accs.avg
        val_loss = valid_losses.avg

        ###########################################################################
        # Log validation metrics to Tensorboard
        ###########################################################################

        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar("RMSE/Val", val_rmse(), epoch)
        writer.add_scalar("RPA/Val", val_rpa(), epoch)
        writer.add_scalar("RCA/Val", val_rca(), epoch)

        ###########################################################################
        # Early stopping
        ###########################################################################

        if val_accuracy - last_val_acc <= 0:
            early_stop_count += 1
        else:
            early_stop_count = 0
            last_val_acc = val_accuracy
        if early_stop_count >= penne.EARLY_STOP_PATIENCE:
            print("Validation accuracy has not improved, stopping early")
            break

        ###########################################################################
        # Checkpoint saving
        ###########################################################################

        if val_loss < best_loss and epoch > 5:
            best_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best.ckpt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, checkpoint_path)
            print("Validation loss improved to " + str(val_loss) + ", best model saved")

        if epoch % penne.CHECKPOINT_FREQ == 0:
            checkpoint_path = checkpoint_dir / (str(epoch) + '.ckpt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, checkpoint_path)
            print("Checkpoint saved at epoch " + str(epoch))

        checkpoint_path = checkpoint_dir / 'latest.ckpt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, checkpoint_path)
        print("Latest model saved")

        # make plots of a specific example every LOG_EXAMPLE_FREQUENCY epochs
        # plot logits and posterior distribution
        if (epoch < 5 or epoch % penne.LOG_EXAMPLE_FREQUENCY == 0):
            # load a batch for logging if not yet loaded
            if ex_batch is None:
                ex_batch = ex_batch_for_logging(penne.LOG_EXAMPLE, device=device)
                if args.harmof0:
                    ex_batch = ex_batch.unsqueeze(0).transpose(2, 1)
            # plot logits
            logits = penne.infer(ex_batch, model=model).cpu()
            # plot posterior distribution
            if penne.LOG_WITH_SOFTMAX:
                write_posterior_distribution(torch.nn.Softmax(dim=1)(logits), writer, epoch)
            write_posterior_distribution(logits, writer, epoch)
            del logits

        val_rmse.reset()
        val_rpa.reset()
        val_rca.reset()

        epoch += 1
    



def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(add_help=False)

    # Add project arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The size of a batch')
    parser.add_argument(
        '--dataset',
        help='The name of the dataset')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number data loading jobs to launch. If None, uses number of ' +
             'cpu cores.')
    parser.add_argument(
        '--name',
        type=str,
        default='training',
        help='The name of the run for logging purposes.')
    parser.add_argument(
        '--pdc',
        action='store_true',
        help='If present, run PDC training')
    parser.add_argument(
        '--gpus',
        type=int,
        help='Device ordinal of GPU to use (-1 for no GPU)',
        default=-1
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=1000,
        help='Max number of epochs to run (if no early stopping)'
    )
    parser.add_argument(
        '--limit_train_batches',
        type=int,
        help='Maximum number of batches to train on',
        default=500
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        help='Maximum number of batches to validate on',
        default=500
    )
    parser.add_argument(
        '--harmof0',
        action='store_true',
        help='Use the HarmoF0 model'
    )
    parser.add_argument(
        '--restart',
        action='store_true',
        help='Restart model training (rather than starting from checkpoint)'
    )

    # Parse
    return parser.parse_args()


if __name__ == '__main__':
    main()
