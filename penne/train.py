"""train.py - model training"""


import argparse
from pathlib import Path

from tqdm import tqdm
import penne
import torch
import torch.nn.functional as F
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
        self.avg = self.sum / self.count

def my_loss(y_hat, y):
        # apply Gaussian blur around target bin
        mean = penne.convert.bins_to_cents(y)
        normal = torch.distributions.Normal(mean, 25)
        bins = penne.convert.bins_to_cents(torch.arange(penne.PITCH_BINS).to(y.device))
        bins = bins[:, None]
        y = torch.exp(normal.log_prob(bins)).permute(1,0)
        y /= y.max(dim=1, keepdims=True).values
        assert y_hat.shape == y.shape
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

def my_acc(y_hat, y):
    argmax_y_hat = y_hat.argmax(dim=1)
    return argmax_y_hat.eq(y).sum().item()/y.numel()

def main():
    """Train a model"""
    # Parse command-line arguments
    args = parse_args()

    # Setup early stopping for 32 epochs (by default according to CREPE) of no val accuracy improvement
    patience = penne.EARLY_STOP_PATIENCE

    # Setup data
    datamodule = penne.data.DataModule(args.dataset,
                                args.batch_size,
                                args.num_workers)
    # early_stop_callback = EarlyStopping(
    #                             monitor='val_accuracy',
    #                             min_delta=0.00,
    #                             patience=patience,
    #                             verbose=False,
    #                             mode='max')

    # Setup log directory and model according to --pdc flag
    if args.pdc:
        logdir = 'pdc'
        model = penne.PDCModel(name=args.name)
    else:
        logdir = 'crepe'
        model = penne.Model(name=args.name)

    #Select device
    if torch.cuda.is_available() and args.gpus > -1:
        device = torch.device('cuda', args.gpus)
    else:
        device = torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=penne.LEARNING_RATE)

    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    
    #If early_stop_count gets above patience, stop early
    early_stop_count = 0
    last_val_acc = 0

    model.train() #Model in training mode
    torch.set_grad_enabled(True)

    best_loss = float('inf')

    train_rmse = penne.metrics.WRMSE()
    train_rpa = penne.metrics.RPA()
    train_rca = penne.metrics.RCA()
    val_rmse = penne.metrics.WRMSE()
    val_rpa = penne.metrics.RPA()
    val_rca = penne.metrics.RCA()

    for epoch in range(1, args.max_epochs + 1):
        train_losses = AverageMeter('Loss', ':.4e')
        train_accs = AverageMeter('Accuracy', ':6.4f')
        valid_losses = AverageMeter('Loss', ':.4e')
        valid_accs = AverageMeter('Accuracy', ':6.4f')
        #Training on each batch (from previous train_step)
        for t, (x, y, voicing) in enumerate(tqdm(train_loader, desc='Epoch ' + str(epoch) + ' training', total=min(len(train_loader), args.limit_train_batches))):
            if t > args.limit_train_batches:
                break
            x, y, voicing = x.to(device), y.to(device), voicing.to(device)
            output = model(x)
            loss = my_loss(output, y)
            acc = my_acc(output, y)

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
            train_accs.update(acc, x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('training loss: %.5f, training accuracy: %.5f' % (train_losses.avg, train_accs.avg))
        train_rmse.reset()
        train_rpa.reset()
        train_rca.reset()

        #Validate on each batch (from previous validation_step)
        with torch.no_grad():
            for t, (x, y, voicing) in enumerate(tqdm(train_loader, desc='Epoch ' + str(epoch) + ' validation', total=min(len(train_loader), args.limit_train_batches))):
                """Performs one step of validation"""
                if t > args.limit_val_batches:
                    break
                x, y, voicing = x.to(device), y.to(device), voicing.to(device)
                output = model(x)
                loss = my_loss(output, y)
                acc = my_acc(output, y)
                
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
                valid_accs.update(acc, x.size(0))
        
        print('validation loss: %.5f, validation accuracy: %.5f' % (valid_losses.avg, valid_accs.avg))
        
        #Check for early stopping
        val_accuracy = valid_accs.avg
        val_loss = valid_losses.avg
        if val_accuracy - last_val_acc <= 0:
            early_stop_count += 1
        else:
            early_stop_count = 0
            last_val_acc = val_accuracy
        if early_stop_count >= penne.EARLY_STOP_PATIENCE:
            print("Validation accuracy has not improved, stopping early")
            break

        #Save the best checkpoint so far
        if val_loss < best_loss and epoch > 5:
            best_loss = val_loss
            cp_path = 'pdc' if args.pdc else 'crepe'
            checkpoint_path = penne.CHECKPOINT_DIR.joinpath(cp_path, args.name, str(epoch)+'.ckpt')
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            print("Validation loss improved to " + str(val_loss) + ", saving to " + str(checkpoint_path))

        val_rmse.reset()
        val_rpa.reset()
        val_rca.reset()

    # Setup tensorboard
    #logger = pl.loggers.TensorBoardLogger(penne.RUNS_DIR / 'logs' / logdir, name=args.name)

    # Setup trainer
    #trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[early_stop_callback])

    # Train
    #trainer.fit(model, datamodule=datamodule)


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
        default=6,
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
        help='Number of GPUs to use (-1 for no GPU)'
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
        help='Maximum number of batches to train on'
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        help='Maximum number of batches to validate on'
    )

    # Add model arguments
    parser = penne.Model.add_model_specific_args(parser)

    # Add trainer arguments
    #parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    return parser.parse_args()


if __name__ == '__main__':
    main()
