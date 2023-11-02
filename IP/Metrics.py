import math
import torch
import numpy as np
import pandas as pd

"""
"""
class Metric(): 
    def __init__(self):
        super().__init__()
    
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor): 
        pass

"""
MemoryCapacity-Tau (Univariate)

@TODO can it be implemented for multivariate signals??
"""
class TauMemoryCapacity(Metric): 
    def evaluate(self, U_tau: torch.Tensor, Y: torch.Tensor):
        # cov = (torch.cov(torch.stack((X_MC, torch.tensor(Y_MC))).view(2,X_MC.shape[0]), correction=0)[0,1])**2
        cov = pd.Series(U_tau.numpy()).cov(pd.Series(Y.numpy()))**2
        return cov/(torch.var(U_tau)*torch.var(Y))
       
"""
Normalized Root of Mean Square Error
"""
class NRMSE(Metric): 
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor):
        return math.sqrt(torch.sum((X - Y)**2)/torch.norm(Y,2))
        
"""
Mean Squared Error 
"""
class MSE(Metric):
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor): 
        #return float(torch.norm(X - Y, 2)/X.shape[0])
        return torch.mean((X - Y)**2) 

"""
Mean Error
"""
class ME(Metric): 
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor): 
        return  torch.mean(X - Y)