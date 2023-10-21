import torch
import numpy as np
from ESN import Reservoir
import matplotlib.pyplot as plt
import torch.nn.functional as F
from IntrinsicPlasticity import IPMask 

class IPReservoir(Reservoir): 
    def __init__(self,  M = 1, N = 10, sparsity=0, ro_rescale = 1, W_range = (-2.5, 2.5), bias = False, bias_range = (-1,1), mask: IPMask = None):  
        super().__init__(M, N, sparsity, ro_rescale, W_range, bias, bias_range)
        
        if mask != None:
            self.set_IP_mask(mask)

        # Initialize the target sample as an empty tensor, so that once a batch of pre training data comes,
        # a tensor with the same number of elements can be sampled from the target distribution.
        self.a = torch.ones(self.N, requires_grad = False)
        self.b = torch.zeros(self.N, requires_grad = False)

        # To evaluate the displacement w.r.t. to the target distribution, KL divergece is the metric. 
        self.kl_log_loss = torch.nn.KLDivLoss(reduction="sum", log_target = True)

        # We will need to sample from the target distribution.
        self.target_sample = torch.tensor([])

        #Initialize an empty buffer where data will be further saved for statistics
        self.buffer = None

    """
    """
    def set_IP_mask(self, mask: IPMask): 
        if self.N != mask.N:
            print(f"Error. Unable to apply a mask with {mask.N} target distributions to a reservoir with {self.N} units.")
            return 
        
        self.mask = mask

    """

    """
    def predict(self,  U: torch.Tensor, save_gradients = False, save_net = False): 
        # Count number of timesteps to be porocessed.
        l = U.shape[0]
        output = torch.zeros((l, self.N))

        if save_net: 
            self.buffer = torch.zeros((l, self.N))

        # Size of U should become [ L X (M + 1) ]
        if self.bias:
          U = torch.column_stack((torch.ones(l), U.float()))

        # Start building computational graph if specified
        # such option will be used when collecting activations for batch IP. 
        if save_gradients == True: 
            self.a.requires_grad = True
            self.b.requires_grad = True

        # Iterate over each input timestamp 
        for i in range(l):
            # Useful to plot neural activity histogram            
            self.net = torch.matmul(torch.mul(U[i], self.W_u), torch.diag(self.a)) + self.b_u + torch.matmul(torch.matmul( self.X, self.W_x), torch.diag(self.a)) + self.b_x + self.b 
            
            if save_net: 
                self.buffer[i, :] = self.net
            
            self.X = self.activation(self.net)
            output[i, :] = self.X

        return output


    """

    """
    def pre_train_batch(self, U: torch.Tensor, eta):
        # If target samples havent't been collected, do it using the target distributions.
        if self.mask == None: 
            print("Error: Unable to train Intrinsic Plasticity without having set any target distribution. Try setting a mask for the reservoir.")
            return 
        
        if self.target_sample.shape[0] == 0:
            timesteps_number = U.shape[0]
            self.mask.sample(timesteps_number)

            if self.mask.apply_activation == True:
                self.target_sample = F.log_softmax(self.activation(self.mask.samples), dim = 1)
            else: 
                self.target_sample = F.log_softmax((self.mask.samples), dim = 1)

            # Save here the evolution of the KL divergence 
            self.loss_history = []
        
        Y = self.activation(self.predict(U, save_gradients=True))
        self.IP_loss = self.kl_log_loss(F.log_softmax(Y, dim = 1), self.target_sample)
        self.loss_history.append(self.IP_loss)

        self.IP_loss.backward(create_graph=True)

        self.a = (self.a - torch.mul(eta, self.a.grad)).clone().detach().requires_grad_(True)
        self.b = (self.b - torch.mul(eta, self.b.grad )).clone().detach().requires_grad_(True)

        self.a.grad = None
        self.b.grad = None



    def pre_train_online(self, U, eta):
        if self.mask.areAllGaussian == False:
            print("WARNING: Only target  Gaussian distributions can be learned online. Use batch IP.")
            return 

        #@TODO implement the analytical online version here.  

    
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
    def printIPstats(self):
        if self.target_sample.shape[0] == 0:
            print("Nothing to print - Reservoir not pretrained yet.")
            return 
        
        if self.buffer == None: 
            print("Nothing to print - No activation saved in the buffer")
            return 
        
        for i in range(self.N):
            actual_std, actual_mean = torch.std_mean(self.buffer[:,i] )
            target_std, target_mean = torch.std_mean(self.mask.samples[:,i])
            print(f"Unit - ({i + 1}): [ ACTUAL_MEAN == ({actual_mean})  ACTUAL_STD == ({actual_std})][ TARGET_MEAN == ({target_mean}) TARGET_STD == ({target_std})]")

        actual_std, actual_mean = torch.std_mean(self.buffer)
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
        if self.buffer == None: 
            print("Nothing to print - No activation saved in the buffer")
            return 

        for i in range(self.buffer.shape[1]):
            x = self.buffer[:,i]

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
    def plot_overall_activation_distribution(self, U, compute_activation = False ):
        if self.buffer == None: 
            print("Nothing to print - No activation saved in the buffer")
            return 
        
        x = self.buffer

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