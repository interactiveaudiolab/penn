# Pitch Estimating Neural NEtworks (PENNE)

## Installation

Clone this repo, navigate to the root directory of the folder and run `pip install -e .`

## Training

### Download data

`python -m penne.data.download`

Downloads and uncompresses the `mdb` and `ptdb` datasets used for training.


### Format data

`python -m penne.data.format`

Converts each dataset to a common format on disk.


### Partition data

`python -m penne.partition`

Generates `train`, `valid`, and `test` partitions for `mdb` and `ptdb`.


### Preprocess data

**TODO** - do we need a preprocess step?

To preprocess data for training, run `python -m penne.preprocess DATASET` where `DATASET` is either `MDB` or `PTDB`.


### Train

**TODO** - dataset -> datasets
**TODO** - better documentation of necessary arguments

`python -m penne.train --dataset=DATASET <args>`

Trains a model. `DATASET` can be `MDB`, `PTDB`, or `BOTH`.
Recommended arguments:
 - `--name=NAME`, which uses NAME for logging organization purposes
 - `--batch_size=32`, which is the batch size used in the original CREPE paper
 - `--limit_train_batches=500` and `--limit_val_batches=500`, which runs 500 random batches per epoch as in the original CREPE paper
 - `--gpus=1` to train on gpu 1. Click [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus) for more info.
 - `--pdc` flag will train using the PDC model


### Monitor

Run tensorboard --logdir runs/. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with ssh -L 6006:localhost:6006 <user>@<server-ip-address>.
Then, open localhost:6006 in your browser.


### Evaluate

**TODO** - dataset -> datasets

```
python -m penne.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>` is the GPU index.
<!--

### Clean

For experiments with cleaner data, you may want to run the clean module.
This allows you to filter out low-scoring stems. Use
`python -m penne.clean <path> <partition> --dataset=DATASET`
where DATASET is either `MDB` or `PTDB`, `<path>` is the path to the per
stem json output from evaluation, and `partition` is `test`, `train`, or `valid`.
This requires you to have already evaluated on the stems in that partition. -->


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A
Convolutional Representation for Pitch Estimation,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP).

[2] J. H. Engel, L. Hantrakul, C. Gu, and A. Roberts,
“DDSP: Differentiable Digital Signal Processing,” in
2020 International Conference on Learning
Representations (ICLR).
