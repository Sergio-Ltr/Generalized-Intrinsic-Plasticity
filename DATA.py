import pandas as pd
import numpy as np
import torch

class NARMA10: 
    def __init__(self, split = True, percentages = [40, 10, 50]):
        data = torch.tensor(pd.read_csv('./NARMA10.csv', header=None).to_numpy())

        self.size = data.shape[1]

        self.X_DATA = data[1,:].reshape((1, self.size))
        self.Y_DATA = data[0,:].reshape((1, self.size))

        if split: 
            self.split(percentages)

    def split(self, percentages=[40,10,50]): 
        prev_idx = 0; 
        X_chunks = []
        Y_chunks = []

        total = sum(percentages)

        for p in percentages: 
            idx = int(p * self.size/total) + prev_idx
            X_chunks.append(self.X_DATA[:,prev_idx: idx])
            Y_chunks.append(self.Y_DATA[:,prev_idx: idx])
            prev_idx

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


