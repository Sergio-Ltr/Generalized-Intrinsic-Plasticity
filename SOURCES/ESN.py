import torch
from scipy.fftpack import fft, ifft
from sklearn.linear_model import Ridge

from Reservoir import Reservoir

"""
Linear reggressor to be attached at the top of a Reservoir, building up an ESN. 
"""
class Readout():
  def __init__(self):
    self.trained = False

  def train(self, X: torch.Tensor, Y: torch.Tensor, lambda_thikonov=0):
    clf = Ridge(alpha=lambda_thikonov)
    clf.fit(X.detach().numpy(), Y.detach().numpy())
    self.clf = clf

    """
    input_size = X.shape[1]
    N = X.shape[0]

    X = torch.column_stack((torch.ones(input_size), X))
    correlation_matrix = torch.matmul(torch.transpose(X,0,1), X)

    if lambda_thikonov != 0:
      correlation_matrix += torch.mul(lambda_thikonov, torch.eye(N+1))

    ## @TODO revrite this
    self.W = torch.matmul(torch.matmul(torch.linalg.inv(correlation_matrix), torch.transpose(X,0,1)), Y.float())
    """
    self.trained = True


  def predict(self, X: torch.Tensor):
    if self.trained:
      # @TODO maybe the shape of output tensor should be transposed.  
      return self.clf.predict(X.detach().numpy())
      #torch.matmul(torch.column_stack((X, torch.ones( X.shape[0]))), self.W)
    else:
      print("Readout not trained - unable to perform any inference.");
      return
    

class EchoStateNetwork():
  def __init__(self, reservoir: Reservoir = Reservoir(1,1,0)):
    self.reservoir = reservoir     
    self.readout = Readout()


  def train(self, U: torch.Tensor, Y: torch.Tensor, lambda_thikonov, transient = 100, verbose = True): 
    if transient != 0: 
      self.reservoir.reset_initial_state()
      warm_up_applied = self.reservoir.warm_up(U[:transient])
      
      if verbose: 
        print(f"Reservoir warmed up with the first {transient} time steps")

      if warm_up_applied:
        U = U[transient:None]
        Y = Y[transient:None]

    X = self.reservoir.predict(U)
    self.readout.train(X, Y, lambda_thikonov) 

    return self.readout.predict(X)

  """

  def evaluate(self, U: torch.Tensor, Y: torch.Tensor, metric: Metric = NRMSE(), plot = False):
    if self.readout.trained == False: 
      return
    
    Y_pred = self.predict(U)

    if plot == True:
      Y_pred_df = pd.DataFrame(Y_pred)
      Y_truth_df = pd.DataFrame(Y.numpy())

      ax = Y_truth_df.plot(grid=True, label='Target', style=['g-','bo-','y^-'], linewidth=0.5, )
      Y_pred_df.plot(grid=True, label='Reconstructed', style=['r--','bo-','y^-'], linewidth=0.5, ax=ax)

    return metric.evaluate( X = Y, Y = Y_pred)
  """

  def predict(self, U:torch.Tensor ): 
    if self.readout.trained == False: 
      return
    
    X = self.reservoir.predict(U)
    return torch.Tensor(self.readout.predict(X)).detach()
  