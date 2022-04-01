# Pitch Estimating Neural NEtworks (PENNE)

## Installation

Clone this repo, navigate to the root directory of the folder and run `pip install -e .`

## Training

### Download data

Place datasets in `data/DATASET`, where `DATASET` is the name of the dataset. `MDB` and `PTDB` are supported. The internal folder hierarchy should be as downloaded ([MDB-stem-synth](https://zenodo.org/record/1481172), [PTDB-TUG](https://www2.spsc.tugraz.at/databases/PTDB-TUG/)), like the following:
```
data
|-- MDB
|   |-- annotation_stems
|   |   |-- ...
|   |-- audio_stems
|   |   |-- ...
|-- PTDB
|   |-- FEMALE
|   |   |-- ...
|   |-- MALE
|   |   |-- ...
```
You can accomplish this manually, or by running the download_data.sh script.

### Partition data

To generate training/testing/validation partitions, run `python -m penne.partition DATASET` where `DATASET` is either `MDB` or `PTDB`.


### Preprocess data

To preprocess data for training, run `python -m penne.preprocess DATASET` where `DATASET` is either `MDB` or `PTDB`.


### Train

To train the model, run `python -m penne.train --dataset=DATASET <args>`. `DATASET` can be `MDB`, `PTDB`, or `BOTH`.
Recommended arguments:
 - `--name=NAME`, which uses NAME for logging organization purposes
 - `--batch_size=32`, which is the batch size used in the original CREPE paper
 - `--limit_train_batches=500` and `--limit_val_batches=500`, which runs 500 random batches per epoch as in the original CREPE paper
 - `--gpus=1` to train on gpu 1. Click [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus) for more info.
 - `--pdc` flag will train using the PDC model


### Monitor

Run `tensorboard --logdir runs/logs`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. Some IDEs (e.g., VS Code) will do this automatically, or
you can do this manually with `ssh -L 6006:localhost:6006 <user>@<server-ip-address>`.
Then, open `localhost:6006` in your browser. By default, CREPE and
PDC trainings are placed in `runs/logs/crepe` and `run/logs/pdc` subfolders
respectively, so you can replace the `--logdir` path to see only those runs
in the tensorboard.


### Evaluate

To evaluate, run `python -m penne.evaluate --dataset=DATASET
--checkpoint=<checkpoint> --model_name=<model_name>`, where
`DATASET` either `MDB` or `PTDB`, `<checkpoint>` is the checkpoint
file to evaluate, and `<model_name>` is a name given to label
this particular evaluation run.`--pdc` flag is required for evaluating
PDC models. Results show up in `runs/eval/DATASET/<model_name>`.


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A
Convolutional Representation for Pitch Estimation,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP).

[2] J. H. Engel, L. Hantrakul, C. Gu, and A. Roberts,
“DDSP: Differentiable Digital Signal Processing,” in
2020 International Conference on Learning
Representations (ICLR).
