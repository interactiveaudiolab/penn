"""train.py - model training"""


import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import penne


###############################################################################
# Train
###############################################################################


def main():
    """Train a model"""
    # Parse command-line arguments
    args = parse_args()

    # Setup tensorboard
    logger = pl.loggers.TensorBoardLogger(penne.RUNS_DIR / 'logs', name=args.name)

    # Setup data
    datamodule = penne.data.DataModule(args.dataset,
                                  args.batch_size,
                                  args.num_workers)

    # Setup early stopping for 32 epochs of no val accuracy improvement
    patience = penne.EARLY_STOP_PATIENCE if not penne.ORIGINAL_CREPE else 32
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='max'
    )

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[early_stop_callback])

    # Train
    trainer.fit(penne.Model(name=args.name), datamodule=datamodule)


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

    # Add model arguments
    parser = penne.Model.add_model_specific_args(parser)

    # Add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    return parser.parse_args()


if __name__ == '__main__':
    main()
