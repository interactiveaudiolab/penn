# Pitch Estimating Neural NEtworks (PENNE)
<!-- [![PyPI](https://img.shields.io/pypi/v/penne.svg)](https://pypi.python.org/pypi/penne) -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://pepy.tech/badge/penne)](https://pepy.tech/project/penne) -->

## Table of contents

- [Installation](#installation)
- [Inference](#inference)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#reproducing-results)
    * [Evaluate](#evaluate)
    * [Analyze](#analyze)
    * [Plot](#plot)
- [Citation](#citation)


## Installation

If you just want to perform pitch estimation using a pretrained model, run
`pip install penne`

If you want to train or use your own models, clone this repo, navigate to the
root directory of the folder and run `pip install -e .`


## Inference

**TODO - example**

**TODO - API**

**TODO - CLI**


## Training

### Download

`python -m penne.data.download`

Downloads and uncompresses the `mdb` and `ptdb` datasets used for training.


### Preprocess

`python -m penne.data.preprocess --config <config>`

Converts each dataset to a common format on disk ready for training. You
can optionally pass a configuration file to override the default configuration.


### Partition

`python -m penne.partition`

Generates `train`, `valid`, and `test` partitions for `mdb` and `ptdb`.
Partitioning is deterministic given the same random seed. You do not need to
run this step, as the original partitions are saved in
`penne/assets/partitions`.


### Train

`python -m penne.train --config <config> --gpus <gpus>`

Trains a model according to a given configuration on the `mdb` and `ptdb`
datasets. Uses a list of GPU indices as an argument, and uses distributed
data parallelism (DDP) if more than one index is given. For example,
`--gpus 0 3` will train using DDP on GPUs `0` and `3`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with `ssh -L 6006:localhost:6006 <user>@<server-ip-address>`.
Then, open `localhost:6006` in your browser.


## Evaluation

### Evaluate


```
python -m penne.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


### Analyze

`python -m penne.evaluate.analyze`

Aggregate model evaluations to produce tables of results.


### Plot

```
python -m penne.plot.density \
    --config <config> \
    --true_datasets <true_datasets> \
    --inference_datasets <inference_datasets> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the true positives of a model on a dataset overlayed on the data
distribution of that dataset and save to a jpg file.

```
python -m penne.plot.inference \
    --config <config> \
    --audio_file <audio_file> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the monophonic pitch contour of an audio file and save to a jpg file.

```
python -m penne.plot.logits \
    --config <config> \
    --audio_file <audio_file> \
    --output_file <output_file> \
    --pitch_file <pitch_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the pitch posteriorgram of an audio file with optional pitch overlay and
save to a jpg file.

```
python -m penne.plot.thresholds \
    --config <config> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the periodicity performance (voiced/unvoiced F1) over mdb and ptdb as a
function of the voiced/unvoiced threshold.


## Citation

### IEEE
M. Morrison, C. Hseih, N. Pruyne, and B. Pardo, "Cross-domain Neural Pitch and Periodicity Estimation," Submitted to <conference>, <month> 2023.


### BibTex

```
@inproceedings{morrison2023cross,
    title={Cross-domain Neural Pitch and Periodicity Estimation},
    author={Morrison, Max and Hseih, Caedon and Pruyne, Nathan and Pardo, Bryan},
    booktitle={Submitted to TODO},
    month={TODO},
    year={2023}
}
