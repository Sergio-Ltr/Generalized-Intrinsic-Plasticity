import torch
import numpy as np
from ESN import Reservoir
import matplotlib.pyplot as plt
import torch.nn.functional as F
from IntrinsicPlasticity import IPMask 

class IPReservoir(Reservoir): 
    def __init__(self,  M = 1, N = 10,  ro_rescale = 1, bias=False):  
        super().__init__(M, N, ro_rescale, bias)
        self.mask = None
       
    """
    @ TODO change the way autoiff check is applied. Controls can be all brought to IPMask class, having the reservoir only applying 
    the eventual analytical learning rule. 
    """    
    def setIPTargets(self, mask: IPMask = None, autodiff = True): 
        if self.N !=  mask.N :
            print("Invalid mask for the numeber of units in the Reservoir.")
        else: 
            # To make the reservoir pretrainable we should consider the target distributions of its units. 
            self.mask = mask if mask != None else IPMask.normalMask(self.N)

            # We need automatic differentiation only if we use a non-gaussian target distribution.
            self.use_autodiff = False if autodiff == False and mask.areAllGaussian == True else True 

            if self.use_autodiff == True: 
                # Initialize the target sample as an empty tensor, so that once a batch of pre training data comes,
                # a tensor with the same number of elements can be sampled from the target distribution.
                self.a = torch.ones(self.N, requires_grad = True)
                self.b = torch.zeros(self.N, requires_grad = True)
            else:
                # If target distribution is gaussian we can avoid gradient computation and use analytical learning rules. 
                self.a = torch.ones(self.N, requires_grad = False)
                self.b = torch.zeros(self.N, requires_grad = False)
        
            # To evaluate the displacement w.r.t. to the target distribution, KL divergece is the metric. 
            self.kl_log_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target = True)

            # We will need to sample from the target distribution.
            self.target_sample = torch.tensor([])


    """

    """
    def shape(self):
        # Call parent method
        super().shape(self)

        # Also print shapes of the IP parameters. 
        print("IP gain", self.a.shape )
        print("IP bias", self.b.shape )


    """

    """
    def predict(self,  U, reset_internal_state = True, save_activation = True, pre_train_IP = False, eta = 0.05): 
        output = super().predict(U, reset_internal_state, save_activation)

        if pre_train_IP:
            self.preTrainIP(eta)
            print(self.IP_loss)

        return output


    """

    """
    def preTrainIP(self, eta):
        # If target samples havent't been collected, do it using the target distributions.
        if self.target_sample.shape[0] == 0:
            timesteps_number = self.activation_buffer.shape[0]
            self.mask.sample(timesteps_number)

            if self.mask.apply_activation == True:
                self.target_sample = F.log_softmax(self.activation(self.mask.samples), dim = 1)
            else: 
                self.target_sample = F.log_softmax((self.mask.samples), dim = 1)

            # Save here the evolution of the KL divergence 
            self.loss_history = []
        
        self.IP_loss = self.kl_log_loss(F.log_softmax(self.activation(self.activation_buffer), dim = 1), self.target_sample)
        self.loss_history.append(self.IP_loss)
        
        print(self.a,self.b)

        if self.use_autodiff == True:
            self.IP_loss.backward(create_graph=True)

            self.a = (self.a - torch.mul(eta, self.a.grad)).clone().detach().requires_grad_(True)
            self.b = (self.b - torch.mul(eta, self.b.grad )).clone().detach().requires_grad_(True)

            self.a.grad = None
            self.b.grad = None
        else:
            delta_b = 0
            delta_a = 0

        print(self.a,self.b)

    """
    """
    def printIPstats(self):
        if self.target_sample.shape[0] == 0:
            print("Nothing to print - Reservoir not pretrained yet.")
        else:
            for i in range(self.N):
                actual_std, actual_mean = torch.std_mean(self.activation_buffer[:,i] )
                target_std, target_mean = torch.std_mean(self.mask.samples[:,i])
                print(f"Unit - ({i + 1}): [ ACTUAL_MEAN == ({actual_mean})  ACTUAL_STD == ({actual_std})][ TARGET_MEAN == ({target_mean}) TARGET_STD == ({target_std})]")

            actual_std, actual_mean = torch.std_mean(self.activation_buffer)
            print(f"Overall network: [ACTUAL_MEAN == ({actual_mean})  ACTUAL_STD == ({actual_std})]")


    def printLossCurve(self): 
        return self.loss_history
    
    def print_eigs(self):

        print("Eigenvalues of the non scaled weights") 
        super().print_eigs()

        print("Eigenvalues of the scaled weights")
        print(torch.view_as_real(torch.linalg.eigvals(torch.matmul(self.W_x,  torch.diag(self.a)))))

    """
    
    """
    def plot_local_neural_activity(self,  compute_activation = False):
        for i in range(self.activation_buffer.shape[1]):
            x = self.activation_buffer[:,i]

            if compute_activation == True:
                x = self.activation(x)

            x = x.detach().numpy()
            y = self.mask.samples[:,i].detach().numpy()

            xs = np.linspace(y.min(), y.max(), 200)
            ys = np.zeros_like(xs)


            #plt.set_title(f"Activations of neuron {i+1}")
            plt.plot(xs, ys)
            plt.hist([x, y], bins="fd", label=['Actual', 'Target'])
            plt.show()


    """
    
    """
    def plot_overall_activation_distribution(self, compute_activation = False ):
        x = self.activation_buffer

        if compute_activation == True:
            x = self.activation(x)

        x = x.flatten().detach().numpy()
        y = self.mask.samples.flatten().detach().numpy()

        xs = np.linspace(y.min(), y.max(), 200)
        ys = np.zeros_like(xs)

        #plt.set_title(f"Activations of neuron {i+1}")
        plt.plot(xs, ys)
        plt.hist([x, y],  bins="fd", label=['Actual', 'Target'])
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()