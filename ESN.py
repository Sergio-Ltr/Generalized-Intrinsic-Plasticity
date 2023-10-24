import torch
import pandas as pd
from Metrics import Metric, NRMSE, MemoryCapacityTau

class Reservoir():
    """
    
    """
    def __init__(self, M=1, N=10, sparsity=0, ro_rescale = 1, W_range = (-2.5, 2.5), bias = True, bias_range = (-1,1)):
        # Number of input features
        self.M = M
        
        # Number of recurrent units
        self.N = N
        
        # Presence of libear bias for input data 
        self.bias = bias

        # Sample recurrent weights from a uniform distribution.
        unitary_dist = torch.distributions.uniform.Uniform(-1, 1)
        # Regulate sparsity using the dropout function (weird but effective)
        self.W_x = torch.nn.functional.dropout(unitary_dist.sample((N, N)), sparsity) 

        # Sample input weights randomly (linear bias are absorbed inside here). 
        w_dist = torch.distributions.uniform.Uniform(W_range[0], W_range[1])
        self.W_u = torch.nn.functional.dropout(w_dist.sample((M, N)), 0)

        if bias: 
          bias_dist = torch.distributions.uniform.Uniform(bias_range[0], bias_range[1])
          self.b_u = bias_dist.sample((1,N))
          self.b_x = bias_dist.sample((1,N))
        else: 
          self.b_u = torch.zeros((1,N))
          self.b_x = torch.zeros((1,N))
          #self.W_u = torch.cat((bias_dist.sample((1,N)), self.W_u))
        
        # Rescale recurrent weights to unitry spectral radius 
        max_eig = max(abs(torch.linalg.eigvals(self.W_x)))
        self.W_x = (self.W_x/max_eig ) * ro_rescale

        # Initialize first internal states with zeros.
        self.Y = torch.zeros(N)
        self.X = torch.zeros(N)

        self.activation = torch.nn.Tanh()


    """
    - U : a L X M tensor representing a timeseries, where L is the number of timesteps and M the number of features.  
    """
    def predict(self, U: torch.Tensor):
        # Count number of timesteps to be porocessed.
        l = U.shape[0]
        output = torch.zeros((l, self.N))

        # Size of U should become L X M + 1
        #if self.bias:
          #U = torch.column_stack((torch.ones(l), U.float()))

        # Iterate over each input timestamp 
        for i in range(l):
            self.Y = self.activation(torch.mul(U[i], self.W_u) + self.b_u + torch.matmul(self.Y, self.W_x) + self.b_x)
            output[i, :] = self.Y

        return output

    """
    """
    def warm_up(self, U:torch.Tensor): 
       if self.Y.any(): 
          print('No transient applied. Reservoir was already warmed up') 
          return False
       
       self.predict(U)
       return True

      
    """
    """
    def reset_initial_state(self): 
       self.Y = torch.zeros(self.N)

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

  def train(self, X: torch.Tensor, Y: torch.Tensor, lambda_thikonov=0):
    input_size = X.shape[0]
    N = X.shape[1]

    X = torch.column_stack((torch.ones(input_size), X))
    correlation_matrix = torch.matmul(torch.transpose(X,0,1), X)

    if lambda_thikonov != 0:
      correlation_matrix += torch.mul(lambda_thikonov, torch.eye(N+1))

    ## @TODO revrite this
    self.W = torch.matmul(torch.matmul(torch.linalg.inv(correlation_matrix), torch.transpose(X,0,1)), Y.float())
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

  def train(self, U: torch.Tensor, Y: torch.Tensor, lambda_thikonov, transient = 100): 
    if transient != 0: 
      warm_up_applied = self.reservoir.warm_up(U[0:transient])

      if warm_up_applied:
        U = U[transient:None]
        Y = Y[transient:None]  

      print(Y.shape)

    X = self.reservoir.predict(U)
    self.readout.train(X, Y, lambda_thikonov) 

  def evaluate(self, U: torch.Tensor, Y: torch.Tensor, metric: Metric = NRMSE(), plot = False):
    if self.readout.trained == False: 
      return
    
    X = self.reservoir.predict(U)
    Y_pred: torch.Tensor = self.readout.predict(X)

    if plot == True:
      Y_pred_df = pd.DataFrame(Y_pred.detach().numpy())
      Y_truth_df = pd.DataFrame(Y.numpy())

      ax = Y_truth_df.plot(grid=True, label='Target', style=['g-','bo-','y^-'], linewidth=0.5, )
      Y_pred_df.plot(grid=True, label='Reconstructed', style=['r--','bo-','y^-'], linewidth=0.5, ax=ax)

    return metric.evaluate( X = Y, Y = Y_pred)

  
  def MC(self, U: torch.Tensor, tau_max = 100): 
    tau_max = self.reservoir.N * 2 if tau_max == 0 else tau_max
    mc = 0

    X = self.reservoir.predict(U)[0: None, :]
    ts_len = X.shape[0]
    
    Y = self.readout.predict(X)

    for tau in range(tau_max):
      mc += MemoryCapacityTau().evaluate(U[0:ts_len -  tau],Y[tau: None])

    return mc
     
  """
  @TODO Implement
  """   
  def lyapunov_exponents(): 
    pass 