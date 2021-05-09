import abc

import torch
import numpy as np
import penne
import math

###############################################################################
# Base metric
###############################################################################


class Metric(abc.ABC):

    @abc.abstractmethod
    def __call__(self):
        """Retrieve the value for the metric"""
        pass

    @abc.abstractmethod
    def update(self):
        """Update the metric with one batch"""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the metric"""
        pass


###############################################################################
# Evaluate
###############################################################################


class F1:
    """Batch update F1 score"""

    def __init__(self, thresh_function):
        self.reset()
        self.thresh_function = thresh_function

    def __call__(self):
        """Compute the aggregate rmse, precision, recall, and f1"""
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def update(self, source, target, source_periodicity):
        """Update the precision, recall, and f1"""
        thresh_source = self.thresh_function(torch.from_numpy(source), torch.from_numpy(source_periodicity)).numpy()
        source_voiced = ~np.isnan(thresh_source)
        target_voiced = target != 0
        overlap = source_voiced & target_voiced
        self.true_positives += overlap.sum()
        self.false_positives += (~source_voiced & target_voiced).sum()
        self.false_negatives += (source_voiced & ~target_voiced).sum()

    # def compute_f1(self, source, target):
    #     source_voiced = ~torch.isnan(source)
    #     target_voiced = ~torch.isnan(target)
    #     overlap = source_voiced & target_voiced
    #     true_positives = overlap.sum().item()
    #     false_positives = (~source_voiced & target_voiced).sum().item()
    #     false_negatives = (source_voiced & ~target_voiced).sum().item()

    #     precision = \
    #         true_positives / (true_positives + false_positives)
    #     recall = \
    #         true_positives / (true_positives + false_negatives)
    #     f1 = 2 * precision * recall / (precision + recall)
    #     return f1


    def reset(self):
        """Reset the F1 score"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

class WRMSE:
    """Batch update WRMSE score"""

    def __init__(self):
        self.reset()

    def __call__(self):
        """Compute the aggregate rmse, precision, recall, and f1"""
        return math.sqrt(self.sum / self.count)

    def update(self, source, target, periodicity):
        """Update the precision, recall, and f1"""
        convert_source = penne.convert.frequency_to_cents(torch.from_numpy(source)).numpy()
        convert_target = penne.convert.bins_to_cents(torch.from_numpy(target)).numpy()
        self.sum += (periodicity * (convert_source-convert_target)**2).sum()
        self.count += source.shape[1]

    def reset(self):
        """Reset the WRMSE score"""
        self.count = 0
        self.sum = 0

class RPA:
    """Batch update RPA score"""

    def __init__(self, thresh_function):
        self.reset()
        self.thresh_function = thresh_function

    def __call__(self):
        return self.sum / self.count

    def update(self, source, target, source_periodicity):
        convert_source = penne.convert.frequency_to_cents(torch.from_numpy(source)).numpy()
        convert_target = penne.convert.bins_to_cents(torch.from_numpy(target)).numpy()

        # thresh_source = self.thresh_function(torch.from_numpy(convert_source), torch.from_numpy(source_periodicity)).numpy()
        # voiced = ~np.isnan(thresh_source) & target != 0

        voiced = target != 0
        # diff = thresh_source[voiced] - convert_target[voiced]
        diff = convert_source[voiced] - convert_target[voiced]
        self.sum += (np.abs(diff) < 50).sum()
        self.count += voiced.sum()


    def reset(self):
        """Reset the RPA score"""
        self.count = 0
        self.sum = 0

class RCA:
    """Batch update RCA score"""

    def __init__(self, thresh_function):
        self.reset()
        self.thresh_function = thresh_function

    def __call__(self):
        return self.sum / self.count

    def update(self, source, target, source_periodicity):
        convert_source = penne.convert.frequency_to_cents(torch.from_numpy(source)).numpy()
        convert_target = penne.convert.bins_to_cents(torch.from_numpy(target)).numpy()

        # thresh_source = self.thresh_function(torch.from_numpy(convert_source), torch.from_numpy(source_periodicity)).numpy()
        # voiced = ~np.isnan(thresh_source) & target != 0

        voiced = target != 0
        # diff = thresh_source[voiced] - convert_target[voiced]
        diff = convert_source[voiced] - convert_target[voiced]
        diff[diff > 600] -= 1200
        diff[diff < -600] += 1200
        self.sum += (np.abs(diff) < 50).sum()
        self.count += voiced.sum()

    def reset(self):
        """Reset the RCA score"""
        self.count = 0
        self.sum = 0