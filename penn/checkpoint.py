import shutil

import huggingface_hub
import torch

import penn


###############################################################################
# Checkpoint utilities
###############################################################################


def download(repo='maxrmorrison/fcnf0-plus-plus', file='fcnf0++.pt'):
    """Download pretrained model weights from HuggingFace"""
    # Download model
    file = huggingface_hub.hf_hub_download(repo_id=repo, filename=file)

    # Make sure folder exists
    penn.DEFAULT_CHECKPOINT.parent.mkdir(exist_ok=True, parents=True)

    # Copy to checkpoint directory
    shutil.copyfile(file, penn.DEFAULT_CHECKPOINT)


def latest_path(directory, regex='*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.stem))))
    return files[-1]


def load(checkpoint_path, model, optimizer=None):
    """Load model checkpoint from file"""
    try:

        # Load checkpoint
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    except FileNotFoundError as error:

        # Maybe download
        if checkpoint_path == penn.DEFAULT_CHECKPOINT:
            download()

            # Load checkpoint
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            raise error

    # Restore model weights
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    step = checkpoint_dict['step']

    return model, optimizer, step


def save(model, optimizer, step, file):
    """Save training checkpoint to disk"""
    # Maybe unpack DDP
    if torch.distributed.is_initialized():
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Save
    checkpoint = {
        'step': step,
        'model': model_state_dict,
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file)
