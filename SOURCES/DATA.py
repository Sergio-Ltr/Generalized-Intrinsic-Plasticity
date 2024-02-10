import pandas as pd
import numpy as np
import torch

class TimeseriesDATA: 
    def __init__(self) -> None:
        self.X_DATA = torch.tensor([])
        self.Y_DATA = torch.tensor([])
        self.size = 0
        pass

    def split(self, percentages=[40,10,50]): 
            prev_idx = 0; 
            X_chunks = []
            Y_chunks = []

            total = sum(percentages)

            for p in percentages: 
                idx = int(p * self.size/total) + prev_idx
                X_chunks.append(self.X_DATA[prev_idx: idx])
                Y_chunks.append(self.Y_DATA[prev_idx: idx])
                prev_idx = idx

            self.X_TR, self.X_VAL, self.X_TS = tuple(X_chunks)
            self.Y_TR, self.Y_VAL, self.Y_TS = tuple(Y_chunks)

    def TR(self):
        return (self.X_TR, self.Y_TR)
    
    def VAL(self):
        return (self.X_VAL, self.Y_VAL)

    def TS(self):
        return (self.X_TS, self.Y_TS)
    
    def X(self): 
        return (self.X_TR, self.X_VAL, self.X_TS)
    
    def Y(self): 
        return (self.Y_TR, self.Y_VAL, self.Y_TS)



class NARMA10(TimeseriesDATA): 
    def __init__(self, split = True, percentages = [40, 10, 50]):
        data = torch.tensor(pd.read_csv('./../../../ESN-IP/DATA/NARMA10.csv', header=None).to_numpy())

        self.size = data.shape[1]

        self.X_DATA = data[1,:]
        self.Y_DATA = data[0,:]

        if split: 
            super().split(percentages)

    
class MG17(TimeseriesDATA): 
    def __init__(self, split = True, percentages = [80, 10, 10]) -> None:
        data = torch.tensor(pd.read_csv('./../../../ESN-IP/DATA/MG17.csv', header=None).T.to_numpy())

        self.size = data.shape[0]
        
        self.X_DATA = data[0: -1,:].flatten()
        self.Y_DATA = data[1: None,:].flatten()

        if split: 
            super().split(percentages)


class UNIFORM(TimeseriesDATA):
    def __init__(self, size=6000, max_delay = 100, range=(-0.8, 0.8), split = True, percentages = [5000, 0, 1000]):
        # super().__init__()
        uniform_dist = torch.distributions.uniform.Uniform(range[0], range[1])
        self.size = size
        self.max_delay = max_delay
        self.X_FULL = uniform_dist.sample((size + max_delay,1)).squeeze()

        self.delay_timeseries(0)

        if split: 
            super().split(percentages)

    def delay_timeseries(self, tau):
        self.tau = tau

        self.Y_DATA = self.X_FULL[0: self.size]
        self.X_DATA = self.X_FULL[tau: self.size + tau] 
 

class VerstraetenDambre(UNIFORM): 
    def __init__(self, p = 1, size=5000, max_delay = 100, split = True, percentages = [80, 0, 20]):
        self.p = p
        range = (-0.8, 0.8)
        
        super().__init__(size, max_delay, range, split, percentages)

    def delay_timeseries(self, tau): 
        R_DATA = [ self.X_FULL[i]*self.X_FULL[i + 1] for i in range(0, self.size)]
        self.Y_DATA = torch.Tensor([np.sign(r)*(np.abs(r)**self.p) for r in R_DATA])
        self.X_DATA = self.X_FULL[tau - 1: self.size + tau - 1] 


class InubushiErgodic(UNIFORM):
    def __init__(self, size=5000, ni=1, max_delay = 100, range=(-0.8, 0.8), split = True, percentages = [80, 10, 10]):
        self.ni = ni
        super().__init__(size, max_delay, range, split, percentages)

    def delay_timeseries(self, tau): 
        self.Y_DATA = torch.Tensor(np.sin(np.array(self.X_FULL[0: self.size])*self.ni))
        self.X_DATA = self.X_FULL[tau - 1: self.size + tau - 1]     


