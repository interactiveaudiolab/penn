import torch


###############################################################################
# Checkpoint utilities
###############################################################################


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
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Restore model
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    step = checkpoint_dict['step']

    return model, optimizer, step


def save(model, optimizer, step, file):
    """Save training checkpoint to disk"""
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file)
