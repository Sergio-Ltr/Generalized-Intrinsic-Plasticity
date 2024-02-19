from Reservoir import Reservoir
import numpy as np

class IPHistory():
    def __init__(self, metrics: list = []): 
        self.metrics = metrics
        self.metric_names = [m.name for m in metrics]
        self.history = []
        pass

    def update(self, model: Reservoir):
        self.history.append(np.array([m.evaluate(model) for m in self.metrics]))

    def printTrainingCurve(self, metric_index): 
        #@TODO implement.
        return #self.loss_history