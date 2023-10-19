import torch
import pandas as pd
from Metrics import Metric, NRMSE, MemoryCapacityTau

class Reservoir():
    """
    
    """
    def __init__(self, M=1, N=10, ro_rescale = 1, bias = True):
        # Number of recurrent units
        self.N = N
        
        # Number of input features
        self.M = M

        # Presence of libear bias for input data 
        self.bias = bias

        # Sample recurrent weights from a uniform distribution.
        w_dist = torch.distributions.uniform.Uniform(-1, 1)
        self.W_x = w_dist.sample((N, N)) 

        # Sample input weights randomly (linear bias are absorbed inside here). 
        self.W_u = w_dist.sample((M + 1 if bias else M, N)) 
        
        # Rescale recurrent weights to unitry spectral radius 
        max_eig = max(abs(torch.linalg.eigvals(self.W_x)))
        self.W_x = (self.W_x/max_eig ) * ro_rescale

        # Matrices then need to be transpoed.
        #self.W_u = torch.transpose(self.W_u , 0, 1)
        #self.W_x = torch.transpose(self.W_x, 0, 1)

        # Initialize first internal states with zeros.
        self.X = torch.zeros(N)
        self.activation = torch.nn.Tanh()


    """
    
    """
    def predict(self, U: torch.Tensor, reset_internal_state = True, save_activation = True):
        if type(self).__name__ == Reservoir.__name__: 
            self.a = torch.ones(self.N, requires_grad = False)
            self.b = torch.zeros(self.N, requires_grad = False)

        # Count number of timesteps to be porocessed.
        input_len = U.shape[1]
        output = torch.zeros((input_len, self.N))

        # Transpose input so that it can be multiplied by the weight matrix. 
        U = torch.transpose(U.float(), 0, 1)

        if self.bias:
          U = torch.column_stack((U, torch.ones(input_len)))

        # Reset Reservoir internal state if specified. 
        if reset_internal_state:
            self.X = torch.zeros(self.N)

        # Prepare a buffer to save internal states if required. 
        if save_activation:
            self.activation_buffer = torch.zeros((input_len, self.N))

        # Iterate over each input timestamp 
        for i in range(input_len):

            self.net = torch.matmul(torch.matmul(U[i,:], self.W_u), torch.diag(self.a)) + torch.matmul(torch.matmul( self.X, self.W_x), torch.diag(self.a)) + self.b
            output[i, :] = self.activation(self.net)

            # Useful to plot neural activity histogram            
            if save_activation:
                self.activation_buffer[i, :] = torch.matmul(torch.matmul(U[i,:], self.W_u), torch.diag(self.a)) + torch.matmul(torch.matmul( self.X, self.W_x), torch.diag(self.a)) + self.b
    
            self.X = self.activation(self.net)

        return output


    """
    
    """
    def shape(self):
        print("Expected input shape", self.N)
        print("Reservoir units", self.M)
        print("Input weights", self.W_u.shape )
        print("Reccurent weights", self.W_x.shape )


    """
    
    """
    def print_eigs(self):
        print(torch.view_as_real(torch.linalg.eigvals(self.W_x)))



class Readout():
  def __init__(self):
    self.trained = False

  def train(self, X, Y, lambda_thikonov=0):
    input_size = X.shape[0]
    N = X.shape[1]

    A = torch.column_stack((X, torch.ones(input_size)))
    correlation_matrix = torch.matmul(torch.transpose(A,0,1), A)

    if lambda_thikonov != 0:
      correlation_matrix += torch.mul(lambda_thikonov, torch.eye(N+1))

    ## @TODO revrite this
    self.W = torch.matmul(torch.matmul(torch.linalg.inv(correlation_matrix), torch.transpose(A,0,1)), torch.transpose(Y.float(), 0,1))
    self.trained = True


  def predict(self, X: torch.Tensor):
    if self.trained:
      # @TODO maybe the shape of output tensor should be transposed.  
      return torch.matmul(torch.column_stack((X, torch.ones( X.shape[0]))), self.W)
    else:
      print("Readout not trained - unable to perform any inference.");
      return
    

class EchoStateNetwork():
  def __init__(self, reservoir: Reservoir = Reservoir(1,1,0)):
    self.reservoir = reservoir     
    self.readout = Readout()

  def train(self, U: torch.Tensor, Y: torch.Tensor, lambda_thikonov, transient = 0): 
    X = self.reservoir.predict(U, True)
    self.readout.train(X, Y, lambda_thikonov) 

  def evaluate(self, U: torch.Tensor, Y: torch.Tensor, metric: Metric = NRMSE(), plot = False):
    if self.readout.trained == False: 
      return
    
    X = self.reservoir.predict(U, False)
    Y_pred: torch.Tensor = torch.transpose(self.readout.predict(X),0, 1) 

    if plot == True:
      Y_pred_df = pd.DataFrame(Y_pred.detach().numpy()).T
      Y_truth_df = pd.DataFrame(Y.numpy()).T

      ax = Y_pred_df.plot(grid=True, label='Reconstructed', style=['r*-','bo-','y^-'], linewidth=2.0)
      Y_truth_df.plot(color='green', grid=True, label='Original', linewidth=0.75, ax=ax)


    return metric.evaluate( X = Y, Y = Y_pred)

  
  def MC(self, U: torch.Tensor, tau_max = 100): 
    tau_max = self.reservoir.N * 2 if tau_max == 0 else tau_max
    mc = 0

    X = self.reservoir.predict(U, True)[0: None, :]
    ts_len = X.shape[0]
    
    Y = torch.transpose(self.readout.predict(X), 0, 1)

    for tau in range(tau_max):
      mc += MemoryCapacityTau().evaluate(U[:, 0:ts_len -  tau],Y[:, tau: None])

    return mc
     
  """
  @TODO Implement
  """   
  def lyapunov_exponents(): 
    pass 