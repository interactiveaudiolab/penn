import torch

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
        self.accuracy = Accuracy()
        self.f1 = F1()
        self.loss = Loss()
        self.pitch_metrics = PitchMetrics()

    def __call__(self):
        return (
            self.accuracy() |
            self.f1() |
            self.loss() |
            self.pitch_metrics())

    def update(self, logits, bins, target, voiced):
        # Detach from graph
        logits = logits.detach()

        # Update loss
        self.loss.update(logits[:, :penn.PITCH_BINS], bins.T)

        # Decode bins, pitch, and periodicity
        with penn.time.timer('decode'):
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
        return self.l1() | self.rca() | self.rmse() | self.rpa()

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


class Accuracy:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'accuracy': (self.true_positives / self.count).item()}

    def update(self, predicted, target):
        self.true_positives += (predicted == target).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.true_positives = 0
        self.count = 0


class F1:

    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = sorted(list(set(
                [2 ** -i for i in range(1, 11)] +
                [.1 * i for i in range(10)])))
        self.thresholds = thresholds
        self.precision = [Precision() for _ in range(len(thresholds))]
        self.recall = [Recall() for _ in range(len(thresholds))]

    def __call__(self):
        result = {}
        iterator = zip(self.thresholds, self.precision, self.recall)
        for threshold, precision, recall in iterator:
            precision = precision()['precision']
            recall = recall()['recall']
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


class L1:
    """L1 pitch distance in cents"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'l1': (self.sum / self.count).item()}

    def update(self, predicted, target):
        self.sum += torch.abs(penn.cents(predicted, target)).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0.


class Loss():

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, logits, bins):
        self.total += penn.train.loss(logits, bins)
        self.count += bins.shape[0]

    def reset(self):
        self.count = 0
        self.total = 0.


class Precision:

    def __init__(self):
        self.reset()

    def __call__(self):
        precision = (
            self.true_positives /
            (self.true_positives + self.false_positives)).item()
        return {'precision': precision}

    def update(self, predicted, voiced):
        self.true_positives += (predicted & voiced).sum()
        self.false_positives += (predicted & ~voiced).sum()

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0


class Recall:

    def __init__(self):
        self.reset()

    def __call__(self):
        recall = (
            self.true_positives /
            (self.true_positives + self.false_negatives)).item()
        return {'recall': recall}

    def update(self, predicted, voiced):
        self.true_positives += (predicted & voiced).sum()
        self.false_negatives += (~predicted & voiced).sum()

    def reset(self):
        self.true_positives = 0
        self.false_negatives = 0


class RCA:
    """Raw chroma accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rca': (self.sum / self.count).item()}

    def update(self, predicted, target):
        # Compute pitch difference in cents
        difference = penn.cents(predicted, target)

        # Forgive octave errors
        difference[difference > (penn.OCTAVE - THRESHOLD)] -= penn.OCTAVE
        difference[difference < -(penn.OCTAVE - THRESHOLD)] += penn.OCTAVE

        # Count predictions that are within 50 cents of target
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0


class RMSE:
    """Root mean square error of pitch distance in cents"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rmse': torch.sqrt(self.sum / self.count).item()}

    def update(self, predicted, target):
        self.sum += (penn.cents(predicted, target) ** 2).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0.


class RPA:
    """Raw prediction accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rpa': (self.sum / self.count).item()}

    def update(self, predicted, target):
        difference = penn.cents(predicted, target)
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0
