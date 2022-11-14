import penne


###############################################################################
# Plot dataset density vs true positive density
###############################################################################


def to_file(true_datasets, inference_datasets, checkpoint=None, gpu=None):

    true_counts = torch.zeros((penne.PITCH_BINS,))
    inference_counts = torch.zeros((penne.PITCH_BINS,))
