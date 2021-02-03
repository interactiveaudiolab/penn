import abc

import torch


###############################################################################
# Base metric
###############################################################################


class Metric(abc.ABC):

    @abstractmethod
    def __call__(self):
        """Retrieve the value for the metric"""
        pass

    @abstractmethod
    def update(self):
        """Update the metric with one batch"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the metric"""
        pass


###############################################################################
# Evaluate
###############################################################################


class F1:
    """Batch update F1 score"""

    def __init__(self):
        self.reset()

    def __call__(self):
        """Compute the aggregate rmse, precision, recall, and f1"""
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def update(self, source, target):
        """Update the precision, recall, and f1"""
        source_voiced = ~torch.isnan(source)
        target_voiced = ~torch.isnan(target)
        overlap = source_voiced & target_voiced
        self.true_positives += overlap.sum().item()
        self.false_positives += (~source_voiced & target_voiced).sum().item()
        self.false_negatives += (source_voiced & ~target_voiced).sum().item()

    def reset(self):
        """Reset the F1 score"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
