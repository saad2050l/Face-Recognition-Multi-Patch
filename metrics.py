import numpy as np
import torch.nn.functional as F


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class TripletMetric(Metric):
    """
    Metric for the models with the Triplet Loss
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, margin):
        distance_positive = (outputs[0] - outputs[1]).pow(2).sum(1)  # .pow(.5)
        distance_negative = (outputs[0] - outputs[2]).pow(2).sum(1)  # .pow(.5)
        pred = F.relu(distance_negative - distance_positive - margin)
        self.correct += (pred > 0).sum()*1.0
        self.total += outputs[0].size()[0]
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

class ContrastiveMetric(Metric):
    """
    Metric for the models with the Siamese Loss
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, margin):
        pred = (outputs[0] - outputs[1]).pow(2).sum(1)  # .pow(.5)
        pred = (pred > 0.5)
        self.correct += (pred != target).sum()*1.0
        self.total += outputs[0].size()[0]
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'Accuracy'