import torch

"""
  Implementation of a Vanilla Reservoir. 
  Mainly done using PyTorch library.
"""
class Reservoir():
    """
      Initializes all the hyperparameter of a reservoir, mainly sampling random weights. 
    """
    def __init__(self, M=1, N=10, desired_rho = 1, input_scaling = 1, bias = True, Wu_range = (-1, 1), Wh_range = (-1, 1),  bu_range = (-1,1), bh_range = (-1,1), Wu_sparsity=0,  Wh_sparsity=0, activation = torch.nn.Tanh()):
        # Number of input features
        self.M = M
        
        # Number of recurrent units
        self.N = N
        
        # Presence of libear bias for input data 
        self.bias = bias

        # Sample weight
        Wu_dist = torch.distributions.uniform.Uniform(Wu_range[0], Wu_range[1])
        Wh_dist = torch.distributions.uniform.Uniform(Wh_range[0], Wh_range[1])
        
        # Apply sparsity
        self.W_u = torch.nn.functional.dropout(Wu_dist.sample((M, N)), Wu_sparsity) * input_scaling # TODO add feature selection parameter (W-U sparsity)
        self.W_h = torch.nn.functional.dropout(Wh_dist.sample((N, N)), Wh_sparsity) 

        if not bias: 
          bu_dist = torch.distributions.uniform.Uniform(bu_range[0], bu_range[1])
          bh_dist = torch.distributions.uniform.Uniform(bh_range[0], bh_range[1])
          self.b_u = bu_dist.sample((1,N)) * input_scaling
          self.b_x = bh_dist.sample((1,N))
        else: 
          self.b_u = torch.zeros((1,N)) * input_scaling
          self.b_x = torch.zeros((1,N))

        self.activation = activation
        
        # Rescale recurrent weights to unitry spectral radius 
        self.rescale_weights(desired_rho)

        # Initialize first internal states with zeros.
        self.Y = torch.zeros(N)

    """
      Porject a M-dimension input signal within a reservoir, returnin its N-dimensional transformed version. 
    """
    def predict(self, U: torch.Tensor):
        # Count number of timesteps to be porocessed.
        l = U.shape[0]
        output = torch.zeros((l, self.N))

        # Iterate over each input timestamp 
        for i in range(l):
            self.Y = self.activation(torch.mul(U[i], self.W_u) + self.b_u + torch.matmul(self.Y, self.W_h) + self.b_x)
            output[i, :] = self.Y

        return output

    """
      Predics an initial part of an input signal so that the reservoir state is not completely null. 
      No output value is returned.
    """
    def warm_up(self, U:torch.Tensor, force = False, verbose = False): 
      if self.Y.any() and not force: 
          if verbose:
            print('No transient applied. Reservoir was already warmed up') 
          return False

      self.predict(U)
      return 

      
    """
      Reset reservoir state, deleting any memory of signals processed in the past.
      Useful before the computation of some intrinsic metric, or before start collecting transformed signal to train a readout. 
    """
    def reset_initial_state(self): 
      self.Y = torch.zeros(self.N)

    
    """
      Log function to remind reservoir configuration sizes.   
    """
    def shape(self):
        print("Expected input shape", self.N)
        print("Reservoir units", self.M)
        print("Input weights", self.W_u.shape )
        print("Reccurent weights", self.W_h.shape )


    """
      Log function to visualize the maxiumum eigenvalue of the recurrent weight matrix followed by all the (N-1) remianing ones.
    """
    def print_eigs(self):
        print(f"Maximum absolute value of the eignevalue of the reccurent weight matrix: { self.rho()}")
        print(torch.view_as_real(torch.linalg.eigvals(self.W_h)))


    """
      Computes the spectral decompostion of the recurrent weight matrices and returns the maximum eigenvalue, a.k.a. the spectral radius
    """
    def rho(self):
        return max(abs(torch.linalg.eigvals(self.W_h)))
     
  
    """
      Applies a simple scaling procedure to the recurrent weight matrix, so that its spectral radius is equal to a desired value. 
    """
    def rescale_weights(self, desired_rho = 0.96, verbose = False): 
        if verbose:
          print(f"Rescaling reccurent weight from their current spetral radius of {self.rho()} to {desired_rho}")
        self.W_h = (self.W_h/self.rho() ) * desired_rho


"""
  Configuration Object. Created to automatize repeated evaluation and grid search procedures. 
"""
class ReservoirConfiguration: 
    def __init__(self, input_dim = 1, N_units = 100, desired_rho = 0.9, input_scaling= 0.1, bias=True, Wu_range = (-1,1), Wh_range = (-1, 1), 
                 bu_range = (-1, 1), bh_range = (-1, 1), Wu_sparsity=0,  Wh_sparsity=0, activation = torch.nn.Tanh(), name="Vanilla"):
        
        self.name = name

        self.input_dim = input_dim
        self.N_units = N_units
        self.desired_rho = desired_rho
        self.input_scaling = input_scaling
        
        self.bias = bias

        self.Wu_range = Wu_range
        self.Wh_range = Wh_range
        self.bu_range = bu_range
        self.bh_range = bh_range

        self.Wu_sparsity = Wu_sparsity
        self.Wh_sparsity = Wh_sparsity

        self.activation = activation


    def build_up_model(self) -> Reservoir: 
        return Reservoir(self.input_dim, self.N_units, self.desired_rho, self.input_scaling, self.bias, self.Wu_range, self.Wh_range, 
                          self.bu_range, self.bh_range, self.Wu_sparsity, self.Wh_sparsity, self.activation)
    

