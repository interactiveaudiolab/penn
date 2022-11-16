# Pitch Estimating Neural NEtworks (PENNE)

**TODO** - API
**TODO** - Figures
**TODO** - Analysis


## Installation

Clone this repo, navigate to the root directory of the folder and run `pip install -e .`


## Training

### Download data

`python -m penne.data.download`

Downloads and uncompresses the `mdb` and `ptdb` datasets used for training.


### Preprocess data

`python -m penne.preprocess`

Converts each dataset to a common format on disk ready for training.


### Partition data

`python -m penne.partition`

Generates `train`, `valid`, and `test` partitions for `mdb` and `ptdb`.


### Train

`python -m penne.train --config <config> --datasets <datasets> --gpus <gpus>`

Trains a model. **TODO** - args


### Monitor

Run tensorboard --logdir runs/. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with ssh -L 6006:localhost:6006 <user>@<server-ip-address>.
Then, open localhost:6006 in your browser.


### Evaluate

```
python -m penne.evaluate \
    --config <config> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>` is the GPU index.
