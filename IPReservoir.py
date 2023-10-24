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

    def sample_targets(self, timesteps_number, overwrite = False): 
        if self.target_sample.shape[0] == 0 or overwrite:
            self.target_sample = self.mask.sample(timesteps_number)

            if self.mask.apply_activation == True: 
                self.target_sample = self.activation(self.target_sample)

        else:
            print(f"Target were already sampled for a {timesteps_number} steps long timeseries. To sample again (and eventually change samples length), set the param 'overwrite=True'.") 

    """

    """
    def predict(self,  U: torch.Tensor, save_gradients = False, save_X = False): 
        # Count number of timesteps to be porocessed.
        l = U.shape[0]
        output = torch.zeros((l, self.N))

        if save_X: 
            self.buffer = torch.zeros((l, self.N))

        # Size of U should become [ L X (M + 1) ]
        #if self.bias:
          #U = torch.column_stack((torch.ones(l), U.float()))

        # Start building computational graph if specified
        # such option will be used when collecting activations for batch IP. 
        if save_gradients == True: 
            self.a.requires_grad = True
            self.b.requires_grad = True

        # Iterate over each input timestamp 
        for i in range(l):
            # Useful to plot neural activity histogram            
            self.X = torch.matmul(torch.mul(U[i], self.W_u), torch.diag(self.a)) + self.b_u + torch.matmul(torch.matmul( self.X, self.W_x), torch.diag(self.a)) + self.b_x + self.b 
            
            if save_X: 
                self.buffer[i, :] = self.X
            
            self.Y = self.activation(self.X)
            output[i, :] = self.Y

        return output


    """

    """
    def pre_train_batch(self, U: torch.Tensor, eta, epochs= 10, transient = 100, verbose = False):
        # If target samples havent't been collected, do it using the target distributions.
        if self.mask == None: 
            print("Error: Unable to train Intrinsic Plasticity without having set any target distribution. Try setting a mask for the reservoir.")
            return 
        
        if transient != 0: 
            warm_up_applied = self.warm_up(U[0:transient])

            if warm_up_applied:
                U = U[transient:None]

        
        if self.target_sample.shape[0] == 0:
            timesteps_number = U.shape[0]
            self.target_sample = self.mask.sample(timesteps_number)

            if self.mask.apply_activation == True:
                #self.target_sample 
                self.softmax_target_sample = F.log_softmax(self.activation(self.target_sample), dim = 1)
            else: 
                self.softmax_target_sample = F.log_softmax((self.target_sample), dim = 1)

            # Save here the evolution of the KL divergence 
            self.loss_history = []
        
        for e in range(epochs):

            Y = self.activation(self.predict(U, save_gradients=True))
            self.IP_loss = self.kl_log_loss(F.log_softmax(Y, dim = 1), self.softmax_target_sample)
            self.loss_history.append(self.IP_loss)

            self.IP_loss.backward(create_graph=True)

            self.a = (self.a - torch.mul(eta, self.a.grad)).clone().detach().requires_grad_(True)
            self.b = (self.b - torch.mul(eta, self.b.grad )).clone().detach().requires_grad_(True)

            self.a.grad = None
            self.b.grad = None

            if verbose: 
                print(f"- Epoch: {e + 1}) | KL Divergence value: {self.IP_loss}.")


    """"
     - U : t
    """
    def pre_train_online(self, U, eta,  epochs=10, transient = 100, verbose=False):
        if self.mask.areAllGaussian == False:
            print("WARNING: Only target  Gaussian distributions can be learned online. Use batch IP.")
            return 

        if transient != 0: 
            warm_up_applied = self.warm_up(U[0:transient])

            if warm_up_applied:
                U = U[transient:None]

        mu = self.mask.means()
        sigma = self.mask.stds()

        square_sigma = torch.mul(sigma, sigma)

        for e in range(epochs):
            # Iterate over each timestep of the input timeseries
            for U_t in U:
                # Fed the reservoir withn the current timestep of the input timeseries, 
                # in order to update the internal states X and Y before applying the online learnin rule. 
                self.predict(torch.tensor([U_t]),False,False)

                summation = 2*square_sigma - 1 - torch.mul(self.Y, self.Y) + torch.mul(mu, self.Y)

                delta_b = torch.mul(- eta, (torch.div(- mu, square_sigma)) + torch.mul(torch.div(self.Y, square_sigma), summation))
                delta_a = torch.div(eta, self.a) + torch.mul(delta_b, self.X) 

                self.b += delta_b.reshape((self.N))
                self.a += delta_a.reshape((self.N))

            if verbose: 
                print(f" - Epoch: { e + 1}")
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
            target_std, target_mean = torch.std_mean(self.target_sample[:,i])
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
            y = self.target_sample[:,i].detach().numpy()

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
        y = self.target_sample.flatten().detach().numpy()

        xs = np.linspace(y.min(), y.max(), 200)
        ys = np.zeros_like(xs)

        #plt.set_title(f"Activations of neuron {i+1}")
        plt.plot(xs, ys)
        plt.hist([x, y],  bins="fd", label=['Actual', 'Target'])
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()