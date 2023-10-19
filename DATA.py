import pandas as pd
import torch

class NARMA10: 
    def __init__(self):
        self.data = torch.tensor(pd.read_csv('./NARMA10.csv').values)
        self.size = self.data.shape[1]
        self.history = []

    def split(self, percentages=[80,10,10]): 
        prev_idx = 0; 
        chunks = []

        total = sum(percentages)

        for p in percentages: 
            idx = int(p * self.size/total) + prev_idx
            chunks.append(self.data[:,prev_idx: idx])
            prev_idx

        return tuple(chunks)

