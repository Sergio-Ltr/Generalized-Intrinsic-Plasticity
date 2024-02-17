import torch
import torch.distributions as D
import functools
import numpy as np
from enum import Enum

class IPDistributionType(Enum): 
    UNIFORM = "UNIFORM" #Equal probability for every number within a range.
    GAUSSIAN = "GAUSSIAN" #Gaussian with user specified mean and standard deviation.
    BIMODAL = "BIMODAL" #Mixture of gaussian with two peaks at opppsite sign means. 
    RANDOM = "RANDOM" #No constraint to values, hence nothing to be optimized. 

class IPDistribution(): 
    def __init__(self, type: IPDistributionType, params):
        
        if type == IPDistributionType.UNIFORM : 
            # param = [left_bound, right_bound]
            self.dist = D.uniform.Uniform(params[0], params[1])
            self.left_bound = params[0]
            self.right_bound = params[1]

        elif type == IPDistributionType.BIMODAL: 
            # params = ([-mean, mean], [std_1, std_2])
            mix = D.Categorical(torch.ones(2,))
            comp = D.Normal(torch.tensor(params[0]), torch.tensor(params[1]))
            self.dist = D.MixtureSameFamily(mix, comp)

        elif type == IPDistributionType.GAUSSIAN: 
            # params = (mean, std)
            self.dist = D.Normal(torch.tensor(params[0]), torch.tensor(params[1]))
            self.mean = params[0]
            self.std = params[1]
            
        # By default a random distribution is assumed, hence a non IP optimized unit. 
        else: #type == IPDistributionType.RANDOM: 
            self.dist = None
            type = IPDistributionType.RANDOM
            print("Error, distribution type not supported")

        self.type: IPDistributionType = type
            
    def isGaussian(self):
        return self.type == IPDistributionType.GAUSSIAN
    
    def sample(self, timesteps_number, units): 
        if self.type == IPDistributionType.BIMODAL or self.type == IPDistributionType.UNIFORM: 
            return self.dist.sample((units, timesteps_number))
        elif self.isGaussian():
            return self.dist.sample((units, timesteps_number))
        else: 
            return torch.zeros((timesteps_number, units))

    # Those static methods allow to istantiate the distribution without using the enum. 
    @staticmethod
    def Uniform(params = [-1.0, 1.0]):
        return IPDistribution(IPDistributionType.UNIFORM, params)

    @staticmethod   
    def Gaussian(params = [0.0, 0.25]):
        return IPDistribution(IPDistributionType.GAUSSIAN, params)

    @staticmethod
    def Normal():
        return IPDistribution.Gaussian([0.0, 1.0])

    @staticmethod
    def Bimodal(params = ([-0.72, 0.72], [0., 0.07])):  
        return IPDistribution(IPDistributionType.BIMODAL, params)

    @staticmethod
    def Random():
        return IPDistribution(IPDistributionType.RANDOM)


"""
    An intrinsic plasticity mask can be viewed as a vector of target distribution to which optimize output of recurrent units of the reservoir. 

    Using a mask, hence multiple target distributions (or also optimizing only a subset of the reservoir units), it would be possible to study wether 
    their co existance can lead to emergent behaviors and hopefully to something resembling criticality. 
"""
class IPMask:
    def __init__(self, distributions : list[IPDistribution], pre_activaiton = False, to_permute = False, name = "Mask"):
        self.N = len(distributions)
        self.distributions = distributions
        self.pre_activaiton = pre_activaiton 
        self.to_permute = to_permute
        self.name = name

        self.areAllGaussian : bool = functools.reduce(lambda  a, b:  a and b.isGaussian(), self.distributions, True )

    def sample(self, timesteps_number): 
        target_samples = torch.zeros((timesteps_number, self.N))

        for i in range(self.N): 
            target_samples[:,i] = self.distributions[i].sample(timesteps_number, 1) 

        return target_samples
    
    def permute_mask(self, mu_neurons): 
        current_means = self.means()
        current_stds = self.stds()

        for neuron_idx, target_idx in zip(np.argsort(mu_neurons), np.argsort(self.means())): 
            self.distributions[neuron_idx].mean = current_means[target_idx]
            self.distributions[neuron_idx].std = current_stds[target_idx]

        self.to_permute = False

    # Useful to compute gradients on the fly
    def means(self):
        if self.areAllGaussian == False:
            return []
        else: 
            return  torch.tensor(list(map(lambda dist: dist.mean, self.distributions)))
    
    # Useful to compute gradients on the fly
    def stds(self):        
        if self.areAllGaussian == False:
            return []
        else: 
            return  torch.tensor(list(map(lambda dist: dist.std, self.distributions)))

    @staticmethod      
    def normal(N, pre_activaiton = False): 
        return IPMask([IPDistribution.Normal() for _ in range(N)], pre_activaiton, name = f"Normal")

    @staticmethod      
    def gaussian(N, std = 0.25, mu = 0.0, pre_activaiton= False): 
        return IPMask([IPDistribution.Gaussian([mu, std]) for _ in range(N)], 
                      pre_activaiton, name = f"Gaussian - mu: {mu} - std: {std}")

    @staticmethod
    def bimodal(N, std = 0.07, mu = 0.72, pre_activaiton = False):
        return IPMask([IPDistribution.Gaussian( [-mu if i % 2 == 0 else +mu, std]) for i in range(N)], 
                      pre_activaiton, name=f"Bimodal - mu: {mu} - std: {std}")

    @staticmethod
    def trimodal(N, linear_rate=(5/9), std_lin = 0.2, std_bim = 0.07, mu = 0.72, pre_activaiton = False):
        return IPMask([IPDistribution.Gaussian([0.0, std_lin]) if i < N*linear_rate else IPDistribution.Gaussian([-mu if i % 2 == 0 else mu, std_bim]) for i in range(N)], 
                      pre_activaiton, name=f"Trimodal - rate:{linear_rate} - mu: {mu} - std_lin: {std_lin} - std_bim:{std_bim}")

    @staticmethod
    def quadrimodal(N, pre_activaiton = False):
        return IPMask([IPDistribution.Gaussian([-0.52 if i % 2 == 0 else 0.52, 0.07]) if i < N*1/9 else 
                       IPDistribution.Gaussian([-0.39 if i % 2 == 0 else 0.39, 0.14]) if i < N*2/9 else
                       IPDistribution.Gaussian([-0.27 if i % 2 == 0 else 0.27, 0.23]) if i < N*3/9 else
                       IPDistribution.Gaussian([-0.72 if i % 2 == 0 else 0.72, 0.07]) for i in range(N)], 
                       pre_activaiton, name = "Quadrimodal")