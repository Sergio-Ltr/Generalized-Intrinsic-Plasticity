import torch
import numpy as np
from torch.nn.modules import Tanh
from Reservoir import Reservoir, ReservoirConfiguration
import matplotlib.pyplot as plt
import torch.nn.functional as F
from IPMask import IPMask 

"""
    Implementation of a Reservoir supporting an Intrinsic Plasticity mask: each neurnos targets a known output probability distribution,
    and tries to optimize a certain loss (KL divergence) between its actual output and that target distribution.
"""
class IPReservoir(Reservoir): 
    
    """
        Intializes the Reservoir as it were a Vanilla one, IP scale and bias are set respectively to 1 and 0, 
        while if passed, a target distribution mask is set as well. 
    """
    def __init__(self, M=1, N=100, desired_rho = 1, input_scaling = 1, bias = True, bu_scaling = 1, bh_scaling = 1,  Wu_sparsity=0,  Wh_sparsity=0, activation = torch.nn.Tanh(), mask: IPMask = None):  
        # Initialize a Vanilla Reservoir following whgat specified in the constructor.
        super().__init__(M, N, desired_rho, input_scaling, bias, bu_scaling, bh_scaling, Wu_sparsity, Wh_sparsity, activation)
        
        # Initialize the target sample as an empty tensor, so that once a batch of pre training data comes,
        # a tensor with the same number of elements can be sampled from the target distribution.
        self.a = torch.ones(N)
        self.b = torch.zeros(N)

        if mask != None:
            self.set_IP_mask(mask)

    """
        Configure target distributions for each neuron of the model, so that IP can then be adapted.
    """
    def set_IP_mask(self, mask: IPMask, permute = True): 
        if self.N != mask.N:
            print(f"Error. Unable to apply a mask with {mask.N} target distributions to a reservoir with {self.N} units.")
            return 
        
        self.mask = mask
        self.mask.to_permute = permute

        # To evaluate the displacement w.r.t. to the target distribution, KL divergece is the metric. 
        self.kl_loss_func = torch.nn.KLDivLoss(reduction="batchmean", log_target = True)

        #Save oriigianl intialization weights before applying the rescale. 
        self.W_u_init = self.W_u
        self.W_h_init = self.W_h

        # We will need to sample from the target distribution.
        self.target_sample = torch.tensor([])

        # To compute thge KL we will use the log softmax of the sample (Pytorch style). 
        self.softmax_target_sample = torch.tensor([])
        

    """"
        Implementation of the online version of the Intrinsic Plasticity Algorithm.
    """
    def IP_online(self, U, eta = 0.000025,  epochs = 10, transient=100, verbose=False):
        # Check if any target distribution has been defined.
        if self.mask == None: 
            print("Error: Unable to train Intrinsic Plasticity without having set any target distribution. Try setting a mask for the reservoir.")
            return 
        
        # Check if the distribution can be learned analytically or autodiff is strictly needed. 
        if self.mask.areAllGaussian == False:
            print("WARNING: Only target  Gaussian distributions can be learned online. Use batch IP.")
            return 
        
        # Get the target distribution parameters from the mask. 
        mu = self.mask.means()
        var = self.mask.stds()**2

        if self.mask.to_permute: 
            self.reset_initial_state()
            self.warm_up(U[:transient])
            self.mask.permute_mask(torch.mean(self.predict(U[transient:None]),axis=0))

        for _ in range(epochs):
            # Iterate over each timestep of the input timeseries

             # Reset reservoir state and warm it up with the first steps of the timeseries. 
            self.reset_initial_state()
            self.warm_up(U[:transient])
            
            for U_t in U[transient:None]:
                #Process current timestep, updating reservoir internal state.  
                self.predict(torch.tensor([U_t]))

                # Compute IP parameter derivatives
                summation = 2 * var + 1 - self.h_t**2 + torch.mul(mu, self.h_t)

                delta_b = - torch.mul(eta, (torch.div(- mu, var)) + torch.mul(torch.div(self.h_t, var), summation))
                delta_a = torch.div(eta, self.a) - eta*torch.mul(delta_b, self.z_t) 

                # Apply online learning rules. 
                self.b += delta_b.reshape((self.N)) 
                self.a += delta_a.reshape((self.N))

                # Updates reservoir weights according to new IP parameters values
                self.update_IP_params()

            #If logs are being collected, this method saves everything in dedicated variables. 
            self.save_epoch_history(U, verbose)


    """
        Generalizes the effect of the IP parameters on reservoir weights. 
    """
    def update_IP_params(self): 
        # Scale both input and reurrent weihgt matrix according to diagonal of IP gain. 
        a_diag = torch.diag(self.a)
        self.W_u = torch.matmul(self.W_u_init, a_diag )
        self.W_h = torch.matmul(self.W_h_init, a_diag)
        
        # Update lienar bias 
        self.total_bias = self.b_u + self.b_h + self.b


    """
        Samples target value for two purposes: evaluate the KL divergence and 
        graphically show an "ideal" data distribution to compare with the actual one.
    """
    def sample_targets(self, timesteps_number): 
        if self.target_sample.shape[0] == timesteps_number:
            return

        if self.target_sample.shape[0] > timesteps_number:
            self.target_sample = self.target_sample[:timesteps_number]
        else: 
            self.target_sample = self.mask.sample(timesteps_number) 
            
        self.softmax_target_sample = F.log_softmax(self.target_sample, dim = 1)
       
    
    """
        Evaluates the KL divergence on a given input signal w.r.t. to target distribution. 
        @ TODO move this in metrics or in
    """
    def evalaute_KL_loss(self, U: torch.Tensor, transient = 100, is_input = True):
        #Check if there is a right number of samples for the comparison 
        self.sample_targets(U.shape[0]-transient)

        #If the signal is totally external, propagate it through the reservoir.
        if is_input: 
            self.reset_initial_state()
            self.warm_up(U, transient)
            H,Z = self.predict(U, return_z = True)
            X = Z if self.mask.pre_activaiton else H
        else:
            X = U

        self.kl_value = self.kl_loss_func(F.log_softmax(X, dim = 1), self.softmax_target_sample)
        
        return self.kl_value


    """
        Display the distibution of neuron activation and compares it with the target one.
    """
    def plot_neural_activity(self, U: torch.Tensor, transient= 100, plot_target = True, pre_activation = False ):
        if pre_activation and not self.mask.pre_activaiton: 
            print("Pre activation target distribution not available")
            plot_target = False
            Y = torch.tensor([])
            
        if plot_target:
            self.sample_targets(U.shape[0] - transient)
            Y = self.target_sample

            if not pre_activation and self.mask.pre_activaiton: 
                Y = self.activation(Y)

        self.reset_initial_state()
        self.warm_up(U, transient)
        H, Z = self.predict(U, return_z = True)
        X = Z if pre_activation else H

        x = X.flatten().detach().numpy()
        y = Y.flatten().detach().numpy()

        xs = np.linspace(y.min(), y.max(), 500)
        ys = np.zeros_like(xs)

        #plt.set_title(f"Activations of neuron {i+1}")
        plt.plot(xs, ys)
        plt.hist([x, y],  bins="fd", label=['Actual', 'Target'])
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()


    """
    
    """
    def save_epoch_history(self, U_val, verbose = True): 
        # @TODO Implement 
        """
            self.sample_targets(N_train, True)

            # Save here the evolution of the KL divergence 
            self.loss_history = []
        """


        """
            self.IP_loss = self.kl_loss_func(F.log_softmax(self.predict(U_val), dim = 1), self.softmax_target_sample)
            self.kl_value = self.IP_loss.detach().flatten()
            self.loss_history.append(self.kl_value)

            if verbose: 
                print(f"- Epoch: {e + 1}) | KL Divergence value: {self.IP_loss}. | Spectral radius: {self.rho()}")
        """
    
    
    """
        Log function to remind reservoir configuration sizes.   
        For the IP reservoir, also gain and bias are printed.
    """
    def shape(self):
        # Call parent method
        super().shape(self)

        # Also print shapes of the IP parameters. 
        print("IP gain", self.a.shape )
        print("IP bias", self.b.shape )


    """
        Kind of alternative constructor, cloning a Vanilla reservoir and the if passed, 
        applying a target distribution mask  as well. 
    """ 
    @staticmethod
    def clone(original: Reservoir): 
        res = IPReservoir()
        
        res.M = original.M 
        res.N = original.N       
    
        res.W_u = original.W_u
        res.b_u = original.b_u

        res.W_h = original.W_h
        res.b_h = original.b_h
        
        res.total_bias = original.total_bias
        res.activation = original.activation

        if isinstance(original, IPReservoir):
            res.a = original.a
            res.b = original.b

            res.W_u_init = original.W_u_init
            res.W_h_init = original.W_h_init

            res.mask = original.mask
        else: 
            res.a = torch.ones(res.N)
            res.b = torch.zeros(res.N)

            res.W_u_init = original.W_u
            res.W_h_init = original.W_h

        return res


#################################################################
    """
    def analyze_neurons(self, U: torch.Tensor, units = []):

        for i in range(self.N) if len(units) == 0 else units:
            actual_std, actual_mean = torch.std_mean(self.Z_buffer[:,i] )
            target_std, target_mean = torch.std_mean(self.target_sample[:,i])
            print(f"Unit - ({i + 1}): [ ACTUAL_MEAN == ({actual_mean})  ACTUAL_STD == ({actual_std})][ TARGET_MEAN == ({target_mean}) TARGET_STD == ({target_std})]")

        actual_std, actual_mean = torch.std_mean(self.Z_buffer)
        print(f"Overall network: [ACTUAL_MEAN == ({actual_mean})  ACTUAL_STD == ({actual_std})]")
    """

    
class IPReservoirConfiguration(ReservoirConfiguration):  
    def __init__(self, config: ReservoirConfiguration, mask: IPMask, eta = 0.0000025, epochs=10, name="IP Reservoir"):
        
        self.config = config
        self.config.name = name
        self.name = name

        self.M = config.M
        self.N = config.N

        self.mask = mask
        self.eta = eta
        self.epochs = epochs
        self.lambda_thikonv = config.lambda_thikonv


    def build_up_model(self, U_TR, transient = 100, plot=False):
        ip_res = IPReservoir.clone( self.config.build_up_model()) 
        ip_res.set_IP_mask(self.mask)

        ip_res.IP_online(U = U_TR, eta =self.eta, epochs=self.epochs, transient=transient)
        
        if plot(plot = False):
            ip_res.plot_neural_activity(U_TR[:int(len(U_TR)/4)])

        return ip_res
    
    def description(self):
        return f"Target: {self.mask.name} |  Eta: {self.eta} - Epochs: {self.epochs} | Initial state: {self.config.description()}"
    
    
    def set_lambda(self, lambda_thikonov):
       self.lambda_thikonv = lambda_thikonov
       super().set_lambda(lambda_thikonov)