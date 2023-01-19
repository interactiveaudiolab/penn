<h1 align="center">Pitch-Estimating Neural Networks (PENN)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/penn.svg)](https://pypi.python.org/pypi/penn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/penn)](https://pepy.tech/project/penn)

</div>

Training, evaluation, and inference of neural pitch and periodicity estimators in PyTorch. Includes the original code for the paper "Cross-domain Neural Pitch and Periodicity Estimation".


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`penn.from_audio`](#pennfrom_audio)
        * [`penn.from_file`](#pennfrom_file)
        * [`penn.from_file_to_file`](#pennfrom_file_to_file)
        * [`penn.from_files_to_files`](#pennfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#evaluation)
    * [Evaluate](#evaluate)
    * [Plot](#plot)
- [Citation](#citation)


## Installation

If you want to perform pitch estimation using a pretrained FCNF0++ model, run
`pip install penn`

If you want to train or use your own models, clone this repo and run
`pip install -r requirements.txt`


## Inference

Perform inference using FCNF0++

```
import penn

# Load audio at the correct sample rate
audio = penn.load.audio('test/assets/gershwin.wav')

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

# Select a checkpoint to use for inference. The default checkpoint will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = penn.DEFAULT_CHECKPOINT

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
pad = True

# Linearly interpolate unvoiced regions below a periodicity threshold
interp_unvoiced_at = penn.DEFAULT_VOICING_THRESHOLD

# Infer pitch and periodicity
pitch, periodicity = penn.from_audio(
    audio,
    penn.SAMPLE_RATE,
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    checkpoint=checkpoint,
    batch_size=batch_size,
    pad=pad,
    interp_unvoiced_at=interp_unvoiced_at,
    gpu=gpu)
```


### Application programming interface

#### `penn.from_audio`

```
"""Perform pitch and periodicity estimation

Args:
    audio: The audio to extract pitch and periodicity from
    sample_rate: The audio sample rate
    hopsize: The hopsize in seconds
    fmin: The minimum allowable frequency in Hz
    fmax: The maximum allowable frequency in Hz
    checkpoint: The checkpoint file
    batch_size: The number of frames per batch
    pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on

Returns:
    pitch: torch.tensor(
        shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
    periodicity: torch.tensor(
        shape=(1, int(samples // penn.seconds_to_sample(hopsize))))
"""
```


#### `penn.from_file`

```
"""Perform pitch and periodicity estimation from audio on disk

Args:
    file: The audio file
    hopsize: The hopsize in seconds
    fmin: The minimum allowable frequency in Hz
    fmax: The maximum allowable frequency in Hz
    checkpoint: The checkpoint file
    batch_size: The number of frames per batch
    pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on

Returns:
    pitch: torch.tensor(shape=(1, int(samples // hopsize)))
    periodicity: torch.tensor(shape=(1, int(samples // hopsize)))
"""
```


#### `penn.from_file_to_file`

```
"""Perform pitch and periodicity estimation from audio on disk and save

Args:
    file: The audio file
    output_prefix: The file to save pitch and periodicity without extension
    hopsize: The hopsize in seconds
    fmin: The minimum allowable frequency in Hz
    fmax: The maximum allowable frequency in Hz
    checkpoint: The checkpoint file
    batch_size: The number of frames per batch
    pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on
"""
```


#### `penn.from_files_to_files`

```
"""Perform pitch and periodicity estimation from files on disk and save

Args:
    files: The audio files
    output_prefixes: Files to save pitch and periodicity without extension
    hopsize: The hopsize in seconds
    fmin: The minimum allowable frequency in Hz
    fmax: The maximum allowable frequency in Hz
    checkpoint: The checkpoint file
    batch_size: The number of frames per batch
    pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
    interp_unvoiced_at: Specifies voicing threshold for interpolation
    gpu: The index of the gpu to run inference on
"""
```


### Command-line interface

```
python -m penn
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    [-h]
    [--config CONFIG]
    [--output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]]
    [--hopsize HOPSIZE]
    [--fmin FMIN]
    [--fmax FMAX]
    [--checkpoint CHECKPOINT]
    [--batch_size BATCH_SIZE]
    [--pad]
    [--interp_unvoiced_at INTERP_UNVOICED_AT]
    [--gpu GPU]

required arguments:
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
        The audio files to process

optional arguments:
    -h, --help
        show this help message and exit
    --config CONFIG
        The configuration file. Defaults to using FCNF0++.
    --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]
        The files to save pitch and periodicity without extension.
        Defaults to audio_files without extensions.
    --hopsize HOPSIZE
        The hopsize in seconds. Defaults to 0.01 seconds.
    --fmin FMIN
        The minimum frequency allowed in Hz. Defaults to 31.0 Hz.
    --fmax FMAX
        The maximum frequency allowed in Hz. Defaults to 1984.0 Hz.
    --checkpoint CHECKPOINT
        The model checkpoint file. Defaults to ./penn/assets/checkpoints/fcnf0++.pt.
    --batch_size BATCH_SIZE
        The number of frames per batch. Defaults to 2048.
    --pad
        If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
  --interp_unvoiced_at INTERP_UNVOICED_AT
        Specifies voicing threshold for interpolation. Defaults to 0.1625.
    --gpu GPU
        The index of the gpu to perform inference on. Defaults to CPU.
```


## Training

### Download

`python -m penn.data.download`

Downloads and uncompresses the `mdb` and `ptdb` datasets used for training.


### Preprocess

`python -m penn.data.preprocess --config <config>`

Converts each dataset to a common format on disk ready for training. You
can optionally pass a configuration file to override the default configuration.


### Partition

`python -m penn.partition`

Generates `train`, `valid`, and `test` partitions for `mdb` and `ptdb`.
Partitioning is deterministic given the same random seed. You do not need to
run this step, as the original partitions are saved in
`penn/assets/partitions`.


### Train

`python -m penn.train --config <config> --gpus <gpus>`

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
python -m penn.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


### Plot

```
python -m penn.plot.density \
    --config <config> \
    --true_datasets <true_datasets> \
    --inference_datasets <inference_datasets> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the data distribution and inferred distribution for a given dataset and
save to a jpg file.

```
python -m penn.plot.logits \
    --config <config> \
    --audio_file <audio_file> \
    --output_file <output_file> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Plot the pitch posteriorgram of an audio file and save to a jpg file.

```
python -m penn.plot.thresholds \
    --names <names> \
    --evaluations <evaluations> \
    --output_file <output_file>
```

Plot the periodicity performance (voiced/unvoiced F1) over mdb and ptdb as a
function of the voiced/unvoiced threshold. `names` are the plot labels to give
each evaluation. `evaluations` are the names of the evaluations to plot.


## Citation

### IEEE
M. Morrison, C. Hsieh, N. Pruyne, and B. Pardo, "Cross-domain Neural Pitch and Periodicity Estimation," IEEE Transactions on Audio, Speech, and Language Processing, <TODO - month> 2023.


### BibTex

```
@inproceedings{morrison2023cross,
    title={Cross-domain Neural Pitch and Periodicity Estimation},
    author={Morrison, Max and Hsieh, Caedon and Pruyne, Nathan and Pardo, Bryan},
    booktitle={IEEE Transactions on Audio, Speech, and Language Processing},
    month={TODO},
    year={2023}
}
