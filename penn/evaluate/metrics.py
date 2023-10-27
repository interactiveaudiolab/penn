import torch
import torchutil

import penn


###############################################################################
# Constants
###############################################################################


# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.accuracy = torchutil.metrics.Accuracy()
        self.f1 = F1()
        self.loss = Loss()
        self.pitch_metrics = PitchMetrics()

    def __call__(self):
        return (
            {
                'accuracy': self.accuracy(),
                'loss': self.loss()
            } |
            self.f1() |
            self.pitch_metrics())

    def update(self, logits, bins, target, voiced):
        # Detach from graph
        logits = logits.detach()

        # Update loss
        self.loss.update(logits[:, :penn.PITCH_BINS], bins.T)

        # Decode bins, pitch, and periodicity
        with torchutil.time.context('decode'):
            predicted, pitch, periodicity = penn.postprocess(logits)

        # Update bin accuracy
        self.accuracy.update(predicted[voiced], bins[voiced])

        # Update pitch metrics
        self.pitch_metrics.update(pitch, target, voiced)

        # Update periodicity metrics
        self.f1.update(periodicity, voiced)

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
        self.loss.reset()
        self.pitch_metrics.reset()


class PitchMetrics:

    def __init__(self):
        self.l1 = L1()
        self.rca = RCA()
        self.rmse = RMSE()
        self.rpa = RPA()

    def __call__(self):
        return {
            'l1': self.l1(),
            'rca': self.rca(),
            'rmse': self.rmse(),
            'rpa': self.rpa()}

    def update(self, pitch, target, voiced):
        # Mask unvoiced
        pitch, target = pitch[voiced], target[voiced]

        # Update metrics
        self.l1.update(pitch, target)
        self.rca.update(pitch, target)
        self.rmse.update(pitch, target)
        self.rpa.update(pitch, target)

    def reset(self):
        self.l1.reset()
        self.rca.reset()
        self.rmse.reset()
        self.rpa.reset()


###############################################################################
# Individual metrics
###############################################################################


class F1:

    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))
        self.thresholds = thresholds
        self.precision = [
            torchutil.metrics.Precision() for _ in range(len(thresholds))]
        self.recall = [
            torchutil.metrics.Recall() for _ in range(len(thresholds))]

    def __call__(self):
        result = {}
        iterator = zip(self.thresholds, self.precision, self.recall)
        for threshold, precision, recall in iterator:
            precision = precision()
            recall = recall()
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.
            result |= {
                f'f1-{threshold:.6f}': f1,
                f'precision-{threshold:.6f}': precision,
                f'recall-{threshold:.6f}': recall}
        return result

    def update(self, periodicity, voiced):
        iterator = zip(self.thresholds, self.precision, self.recall)
        for threshold, precision, recall in iterator:
            predicted = penn.voicing.threshold(periodicity, threshold)
            precision.update(predicted, voiced)
            recall.update(predicted, voiced)

    def reset(self):
        """Reset the F1 score"""
        for precision, recall in zip(self.precision, self.recall):
            precision.reset()
            recall.reset()


class L1(torchutil.metrics.L1):
    """L1 pitch distance in cents"""
    def update(self, predicted, target):
        super().update(
            penn.OCTAVE * torch.log2(predicted),
            penn.OCTAVE * torch.log2(target))


class Loss(torchutil.metrics.Average):
    """Batch-updating loss"""
    def update(self, logits, bins):
        super().update(penn.loss(logits, bins), bins.numel())


class RCA(torchutil.metrics.Average):
    """Raw chroma accuracy"""
    def update(self, predicted, target):
        # Compute pitch difference in cents
        difference = penn.cents(predicted, target)

        # Forgive octave errors
        difference[difference > (penn.OCTAVE - THRESHOLD)] -= penn.OCTAVE
        difference[difference < -(penn.OCTAVE - THRESHOLD)] += penn.OCTAVE

        # Count predictions that are within 50 cents of target
        super().update(
            (torch.abs(difference) < THRESHOLD).sum(),
            predicted.numel())


class RMSE(torchutil.metrics.RMSE):
    """Root mean square error of pitch distance in cents"""
    def update(self, predicted, target):
        super().update(
            penn.OCTAVE * torch.log2(predicted),
            penn.OCTAVE * torch.log2(target))


class RPA(torchutil.metrics.Average):
    """Raw prediction accuracy"""
    def update(self, predicted, target):
        difference = penn.cents(predicted, target)
        super().update(
            (torch.abs(difference) < THRESHOLD).sum(),
            predicted.numel())
