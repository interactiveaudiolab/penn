import torch

import penne


###############################################################################
# Constants
###############################################################################


# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents

# One octave in cents
OCTAVE = 1200  # cents


###############################################################################
# Evaluate
###############################################################################


class F1:
    """Batch update F1 score"""

    def __init__(self, thresh_function):
        self.reset()
        self.thresh_function = thresh_function

    def __call__(self):
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def update(self, source, target, source_periodicity):
        # use threshold to extract predicted voicings
        thresh_source = self.thresh_function(torch.from_numpy(source), torch.from_numpy(source_periodicity)).numpy()
        source_voiced = ~torch.isnan(thresh_source)
        # get voiced frames from annotation
        target_voiced = target != 0
        # get frames that are voiced for both source and target
        overlap = source_voiced & target_voiced
        # true positive = voiced in source and target
        self.true_positives += overlap.sum()

        self.false_negatives += (~source_voiced & target_voiced).sum()
        self.false_positives += (source_voiced & ~target_voiced).sum()

    def reset(self):
        """Reset the F1 score"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0


class RMSE:
    """Pitch RMSE"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return torch.sqrt(self.sum / self.count)

    def update(self, source, target, voiced):
        self.sum += (cents(source[voiced], target[voiced]) ** 2).sum()
        self.count += voiced.sum()

    def reset(self):
        """Reset the WRMSE score"""
        self.count = 0
        self.sum = 0


class RPA:
    """Raw prediction accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return self.sum / self.count

    def update(self, source, target, voiced):
        difference = cents(source[voiced], target[voiced])
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += voiced.sum()

    def reset(self):
        self.count = 0
        self.sum = 0


class RCA:
    """Raw chroma accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return self.sum / self.count

    def update(self, source, target, voiced):
        # Compute pitch difference in cents
        difference = cents(source[voiced], target[voiced])

        # Forgive octave errors
        difference[difference > (OCTAVE - THRESHOLD)] -= OCTAVE
        difference[difference < -(OCTAVE - THRESHOLD)] += OCTAVE

        # Count predictions that are within 50 cents of target
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += voiced.sum()

    def reset(self):
        self.count = 0
        self.sum = 0
