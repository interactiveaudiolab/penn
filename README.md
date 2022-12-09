# Pitch Estimating Neural NEtworks (PENNE)
[![PyPI](https://img.shields.io/pypi/v/penne.svg)](https://pypi.python.org/pypi/penne)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/penne)](https://pepy.tech/project/penne)

Training, evaluation, and inference of neural pitch and periodicity estimators in PyTorch.


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`penne.from_audio`](#pennefrom_audio)
        * [`penne.from_file`](#pennefrom_file)
        * [`penne.from_file_to_file`](#pennefrom_file_to_file)
        * [`penne.from_files_to_files`](#pennefrom_files_to_files)
    * [Command-line interface](#command-line-interface)
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

```
import penne

# Load audio at the correct sample rate
audio = penne.load.audio('test/assets/gershwin.wav')

# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = 0

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Pick a model to use. One of ['crepe', 'deepf0', 'fcnf0']. Default is 'fcnf0'.
model = 'fcnf0'

# Select a checkpoint to use for inference
checkpoint = penne.DEFAULT_CHECKPOINT

# Infer pitch and periodicity
pitch, periodicity = penne.from_audio(
    audio,
    penne.SAMPLE_RATE,
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    model=model,
    checkpoint=checkpoint,
    batch_size=batch_size,
    gpu=gpu)
```


### Application programming interface

**TODO - sphinx link


### Command-line interface

```
python -m penne
    [-h]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    [--output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]]
    [--hopsize HOPSIZE]
    [--fmin FMIN]
    [--fmax FMAX]
    [--model MODEL]
    [--checkpoint CHECKPOINT]
    [--batch_size BATCH_SIZE]
    [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio file to process
  --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]
                        The files to save pitch and periodicity without extension
  --hopsize HOPSIZE     The hopsize in seconds
  --fmin FMIN           The minimum frequency allowed
  --fmax FMAX           The maximum frequency allowed
  --model MODEL         The name of the estimator model
  --checkpoint CHECKPOINT
                        The model checkpoint file
  --batch_size BATCH_SIZE
                        The number of frames per batch
  --gpu GPU             The index of the gpu to perform inference on
```


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
