import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from sklearn.linear_model import Ridge
from DATA import MC_UNIFORM
from Metrics import Metric, NRMSE, TauMemoryCapacity


class Reservoir():
    """
    
    """
    def __init__(self, M=1, N=10, sparsity=0, ro_rescale = 1, W_range = (-1, 1), 
                 bias = True, bias_range = (-1,1), input_scaling = 1, activation = torch.nn.Tanh()):
        # Number of input features
        self.M = M
        
        # Number of recurrent units
        self.N = N
        
        # Presence of libear bias for input data 
        self.bias = bias

        # Sample recurrent weights from a uniform distribution.
        unitary_dist = torch.distributions.uniform.Uniform(W_range[0], W_range[1])
        # Regulate sparsity using the dropout function (weird but effective)
        self.W_x = torch.nn.functional.dropout(unitary_dist.sample((N, N)), sparsity) 

        # Sample input weights randomly (linear bias are absorbed inside here). 
        w_dist = torch.distributions.uniform.Uniform(W_range[0], W_range[1])
        self.W_u = torch.nn.functional.dropout(w_dist.sample((M, N)), 0) * input_scaling # TODO add feature selection parameter (W-U sparsity)

        if bias: 
          bias_dist = torch.distributions.uniform.Uniform(bias_range[0], bias_range[1])
          self.b_u = bias_dist.sample((1,N)) * input_scaling
          self.b_x = bias_dist.sample((1,N))
        else: 
          self.b_u = torch.zeros((1,N)) * input_scaling
          self.b_x = torch.zeros((1,N))
          #self.W_u = torch.cat((bias_dist.sample((1,N)), self.W_u))
        
        # Rescale recurrent weights to unitry spectral radius 
        self.rescale_weights(ro_rescale)
        #max_eig = max(abs(torch.linalg.eigvals(self.W_x)))
        #self.W_x = (self.W_x/max_eig ) * ro_rescale

        # Initialize first internal states with zeros.
        self.Y = torch.zeros(N)
        self.X = torch.zeros(N)

        self.activation = activation


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
    def warm_up(self, U:torch.Tensor, force = False, verbose = False): 
       if self.Y.any() and not force: 
          if verbose:
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
        print(f"Maximum absolute value of the eignevalue of the reccurent weight matrix: { max(abs(torch.linalg.eigvals(self.W_x)))}")
        print(torch.view_as_real(torch.linalg.eigvals(self.W_x)))


    """

    """
    def max_eigs(self):
        return max(abs(torch.linalg.eigvals(self.W_x)))
    
    
    """
    """
    def rescale_weights(self, ro_rescale = 0.96, verbose = False): 
        if verbose:
          print(f"Rescaling reccurent weight from their current spetral radius of {self.max_eigs()} to {ro_rescale}")
        self.W_x = (self.W_x/self.max_eigs() ) * ro_rescale


    """
    Lyapunov characteristic exponent, computed according to Gallicchio et al. in the paper "Local Lyapunov Exponent of Deep Echo State Networks". 
    """
    def LCE(self, U: torch.Tensor, a = 1, transient = 100):
      self.reset_initial_state()
      self.warm_up(U[0:transient])

      U = U[transient:None]

      eig_acc = 0
      W_rec = self.W_x * a
      N_s = U.shape[0]

      for t in range(N_s):
          self.predict(torch.Tensor(U[t:t+1]))
          D = torch.diag(1 - self.Y**2).numpy()
          eig_k, _ = np.linalg.eig(D*W_rec.numpy())
          eig_acc += np.log(np.absolute(eig_k))

      return max(eig_acc/N_s)
    
    """
    Deviation from linearity, measure proposed by Verstraeten et al. in the paper "Memory versus Non-Linearity in Reservoirs".
    """
    def de_fi(self, verbose = False, plot=False, theta_range=(np.linspace(0.01, 0.5, 100)*200).astype(int), starttime = 0.0, endtime = 2.0, steps = 1000):
      de_acc = 0
      t = np.linspace(starttime, endtime, num=steps)
      
      for theta in theta_range:

        f = np.sin(2*np.pi*theta*t) 

        f_res = self.predict(f).numpy()
        f_res -= np.mean(f_res, axis=0)
        f_res = np.mean(f_res, axis=1)

        fhat = np.fft.fft(f_res)
        N = len(fhat)
        halvedfhat = fhat[0:int(N/2)]
        powspec = abs(halvedfhat)**2

        fs = steps/(endtime - starttime)

        freq = np.linspace(0,int(fs/2),int(N/2))

        de_fi_theta = 1 - powspec[2*theta]/np.sum(powspec)

        if verbose: 
          print(f"Frequence:{theta}, Deviation: {de_fi_theta},  Powerspect: {powspec[2*theta]}, Total Energy: {np.sum(powspec)}") 

        if plot:
          plt.plot(freq,powspec)
          plt.xlim([0,100])

        de_acc += de_fi_theta

      return de_acc/len(theta_range)

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
      warm_up_applied = self.reservoir.warm_up(U[0:transient])
      
      if verbose: 
         print(f"Reservoir warmed up with the first {transient} time steps")

      if warm_up_applied:
        U = U[transient:None]
        Y = Y[transient:None]

    X = self.reservoir.predict(U)
    self.readout.train(X, Y, lambda_thikonov) 

    return self.readout.predict(X)


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


  def predict(self, U:torch.Tensor ): 
    if self.readout.trained == False: 
      return
    
    X = self.reservoir.predict(U)
    return torch.Tensor(self.readout.predict(X)).detach()
  

  def MemoryCapacity(self, l = 6000, tau_max = 100, lambda_thikonov = 0, TR_SIZE = 5000, TS_SIZE = 1000): 
    # Take tau as the double of the Reservoir units, according to IP paper. 
    tau_max = self.reservoir.N * 2 if tau_max == 0 else tau_max 
    data = MC_UNIFORM(l,tau_max)
    mc = 0
    #sigma_U = torch.var(data.X_DATA)

    for tau in range(tau_max):
      
      data.delay_timeseries(tau)
      data.split([TR_SIZE, 0 ,TS_SIZE]) 

      U_TR, Y_TR = data.TR()
      U_TS, Y_TS = data.TS()

      self.train(U_TR, Y_TR, lambda_thikonov, transient=100, verbose=False)

      X_TS = self.predict(U_TS)

      target_mean = np.mean(Y_TS.numpy())
      output_mean = np.mean(X_TS.numpy()) 
      
      num, denom_t, denom_out = 0, 0, 0

      for i in range(TS_SIZE):
          deviat_t = Y_TS[i] - target_mean
          deviat_out = X_TS[i] - output_mean
          num += deviat_t * deviat_out
          denom_t += deviat_t**2
          denom_out += deviat_out**2
      num = num**2
      den = denom_t * denom_out
      mc += num/den

      #MC[k] = num/den
      #mc += TauMemoryCapacity().evaluate(U_TS, Y_TS)/sigma_U

    return mc
  