# Pitch Estimating Neural NEtworks (PENNE)

## Quick Guide

1. Clone this repo
2. Run `cd penne`
3. Run `pip install -e .`
4. Put datasets into the data folder. MDB and PTDB are supported. The internal folder hierarchy should be as downloaded ([MDB-stem-synth](https://zenodo.org/record/1481172), [PTDB-TUG](https://www2.spsc.tugraz.at/databases/PTDB-TUG/)), like the following:
```
penne
|-- data
|   |-- MDB
|   |   |-- annotation_stems
|   |   |   |-- ...
|   |   |-- audio_stems
|   |   |   |-- ...
|   |-- PTDB
|   |   |-- FEMALE
|   |   |   |-- ...
|   |   |-- MALE
|   |   |   |-- ...
```
5. To generate training/testing/validation partitions, run `python -m penne.partition DATASET` where DATASET is either MDB or PTDB
6. To preprocess data for training, run `python -m penne.preprocess DATASET` where DATASET is either MDB or PTDB. You can add flag `--voiceonly=True` to allow for training on only voiced frames
7. Optionally, you can change certain constant in `core.py` to change training settings. `ORIGINAL_CREPE = True` overrides other variables and uses settings from the original CREPE paper.
8. To train the model, run `python -m penne.train --dataset=DATASET <args>`. We recommend the following flags:
    * `--name=NAME`, which uses NAME for logging organization purposes
    * `--batch_size=32`, which is the batch size used in the original CREPE paper
    * `--limit_train_batches=500` and `--limit_val_batches=500`, which runs 500 random batches per epoch as in the original CREPE paper

## Installation

Clone this repo and run `cd penne && pip install -e .`.


## Training

### Download data

Place datasets in `data/DATASET`, where `DATASET` is the name of the dataset.


### Partition data

Complete all TODOs in `partition.py`, then run `python -m penne.partition
DATASET`.


### Train

Complete all TODOs in `data.py` and `model.py`. Then, create a directory in
`runs` for your experiment. Logs, checkpoints, and results should be saved to
this directory. In your new directory, run `python -m penne.train --dataset
DATASET <args>`. See the [PyTorch Lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-flags)
for additional arguments. Model-specific arguments should be added in
`penne.Model.add_model_specific_args`.


### Evaluate

Complete all TODOs in `evaluate.py`, then run `python -m penne.evaluate DATASET
<partition> <checkpoint> <file>`, where `<partition>` is the name of the
partition to evaluate, `<checkpoint>` is the checkpoint file to evaluate, and
`<file>` is the json file to write results to.


### Monitor

Run `tensorboard --logdir runs/<run>/logs`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.
Some IDEs (e.g., VS Code) will do this automatically.


### Test

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
Run tests with the command `pytest`. Adding project-specific tests for
data loading and inference is encouraged.


## Usage

### Computing pitch and periodicity from audio


```python
import penne


# Load audio
audio, sr = penne.load.audio( ... )

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 550

# Select a model capacity--one of "tiny" or "full"
model = 'tiny'

# Choose a device to use for inference
device = 'cuda:0'

# Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 2048

# Compute pitch using first gpu
pitch = penne.predict(audio,
                      sr,
                      hop_length,
                      fmin,
                      fmax,
                      model,
                      batch_size=batch_size,
                      device=device)
```

A periodicity metric similar to the Crepe confidence score can also be
extracted by passing `return_periodicity=True` to `penne.predict`.

By default, `penne` uses Viterbi decoding on the softmax of the network
output. This is different than the original implementation, which uses a
weighted average near the argmax of binary cross-entropy probabilities.
The argmax operation can cause double/half frequency errors. These can be
removed by penalizing large pitch jumps via Viterbi decoding. The `decode`
submodule provides some options for decoding.

```python
# Decode using viterbi decoding (default)
penne.predict(..., decoder=penne.decode.viterbi)

# Decode using weighted argmax (as in the original implementation)
penne.predict(..., decoder=penne.decode.weighted_argmax)

# Decode using argmax
penne.predict(..., decoder=penne.decode.argmax)
```

When periodicity is low, the pitch is less reliable. For some problems, it
makes sense to mask these less reliable pitch values. However, the periodicity
can be noisy and the pitch has quantization artifacts. `penne` provides
submodules `filter` and `threshold` for this purpose. The filter and threshold
parameters should be tuned to your data. For clean speech, a 10-20 millisecond
window with a threshold of 0.21 has worked.

```python
# We'll use a 15 millisecond window assuming a hop length of 5 milliseconds
win_length = 3

# Median filter noisy confidence value
periodicity = penne.filter.median(periodicity, win_length)

# Remove inharmonic regions
pitch = penne.threshold.At(.21)(pitch, periodicity)

# Optionally smooth pitch to remove quantization artifacts
pitch = penne.filter.mean(pitch, win_length)
```

For more fine-grained control over pitch thresholding, see
`penne.threshold.Hysteresis`. This is especially useful for removing
spurious voiced regions caused by noise in the periodicity values, but
has more parameters and may require more manual tuning to your data.


### Computing the model output activations

```python
batch = next(penne.preprocess(audio, sr, hop_length))
probabilities = penne.infer(batch)
```


### Computing the CREPE embedding space

As in Differentiable Digital Signal Processing [2], this uses the output of the
fifth max-pooling layer as a pretrained pitch embedding

```python
embeddings = penne.embed(audio, sr, hop_length)
```

### Computing from files

`penne` defines the following functions convenient for predicting
directly from audio files on disk. Each of these functions also takes
a `device` argument that can be used for device placement (e.g.,
`device='cuda:0'`).

```python
penne.predict_from_file(audio_file, ...)
penne.predict_from_file_to_file(
    audio_file, output_pitch_file, output_periodicity_file, ...)
penne.predict_from_files_to_files(
    audio_files, output_pitch_files, output_periodicity_files, ...)

penne.embed_from_file(audio_file, ...)
penne.embed_from_file_to_file(audio_file, output_file, ...)
penne.embed_from_files_to_files(audio_files, output_files, ...)
```

### Command-line interface

```bash
usage: python -m penne
    [-h]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--hop_length HOP_LENGTH]
    [--output_periodicity_files OUTPUT_periodicity_FILES [OUTPUT_periodicity_FILES ...]]
    [--embed]
    [--fmin FMIN]
    [--fmax FMAX]
    [--model MODEL]
    [--decoder DECODER]
    [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio file to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The file to save pitch or embedding
  --hop_length HOP_LENGTH
                        The hop length of the analysis window
  --output_periodicity_files OUTPUT_periodicity_FILES [OUTPUT_periodicity_FILES ...]
                        The file to save periodicity
  --embed               Performs embedding instead of pitch prediction
  --fmin FMIN           The minimum frequency allowed
  --fmax FMAX           The maximum frequency allowed
  --model MODEL         The model capacity. One of "tiny" or "full"
  --decoder DECODER     The decoder to use. One of "argmax", "viterbi", or
                        "weighted_argmax"
  --gpu GPU             The gpu to perform inference on
```


## Tests

The module tests can be run as follows.

```bash
pip install pytest
pytest
```


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A
Convolutional Representation for Pitch Estimation,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP).

[2] J. H. Engel, L. Hantrakul, C. Gu, and A. Roberts,
“DDSP: Differentiable Digital Signal Processing,” in
2020 International Conference on Learning
Representations (ICLR).
