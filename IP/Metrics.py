import math
import torch
import numpy as np


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
class MemoryCapacityTau(Metric): 
    def evaluate(self, U_tau: torch.Tensor, Y: torch.Tensor):
        corr = torch.corrcoef(torch.stack((U_tau, Y)).reshape(2, Y.shape[0]))
        r =  np.diag(np.fliplr(corr))[0]
        return  r*r
    
       
"""
Normalized Root of Mean Square Error
"""
class NRMSE(Metric): 
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor):
        return math.sqrt(torch.mul(torch.norm(X - Y, 2), 1/torch.norm(Y,2)))
        
"""
Mean Squared Error 
"""
class MSE(Metric):
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor): 
        return float(torch.norm(X - Y, 2)/X.shape[0])

"""
Mean Error
"""
class ME(Metric): 
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor): 
        return  float(torch.norm(X - Y, 1)/X.shape[0])