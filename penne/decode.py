


def argmax(logits):
    """Decode pitch using argmax"""
    # Get pitch bins
    bins = logits.argmax(dim=1)

    # Convert to hz
    pitch = penne.convert.bins_to_frequency(bins)

    return bins, pitch


def weighted(logits):
    """Decode pitch using a normal assumption around the argmax"""
    # TODO
    pass
