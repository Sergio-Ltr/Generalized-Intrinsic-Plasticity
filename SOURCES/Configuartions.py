from Reservoir import ReservoirConfiguration
from IPReservoir import IPReservoirConfiguration
from IPMask import IPMask
from DATA import MG17, TimeseriesDATA
from Evaluator import Evaluator
from Metrics import *
import itertools
import os



class ReservoirHyperparamSpace(): 
    def __init__(self, unit_range=[100, 250], rho_range=[ 0.5, 0.75, 0.95, ], input_scaling_range=[1], bias_switch=[True, False], bu_scaling_range=[1], bh_scaling_range=[1],  h_sparsity_range=[0], u_sparsity_range=[0]) -> None:
        self.unit_range = unit_range
        self.rho_range = rho_range
        self.input_scaling_range = input_scaling_range
        self.bias_switch = bias_switch

        self.bu_scaling_range=bu_scaling_range
        self.bh_scaling_range =bh_scaling_range

        self.h_sparsity_range = h_sparsity_range
        self.u_sparsity_range = u_sparsity_range

        self.bias_space = [[False, 0,0] if switch == False else [True, u,h] for switch,u,h in  itertools.product(*[self.bias_switch, self.bu_scaling_range, self.bh_scaling_range])]
        self.bias_space = list(filter(lambda bias: bias[0] != False, self.bias_space)) + ([[False, 0, 0]] if self.bias_switch[0] ==False or len(self.bias_switch)>1 else [])
    def combinations(self): 
        
        return itertools.product(*[self.unit_range, self.rho_range, self.input_scaling_range, self.bias_space, self.h_sparsity_range, self.u_sparsity_range])
    
    def get_configs(self, M=1): 
        
        return [ReservoirConfiguration(M=M, N=N, desired_rho=rho, input_scaling=input_scaling, bias=bias[0], bu_scaling=bias[1], bh_scaling=bias[2], Wh_sparsity=h_sparsity, Wu_sparsity=u_sparsity) for 
                N, rho, input_scaling, bias, h_sparsity, u_sparsity in self.combinations()]
         

class IpTargetDistributionSpace():
    def __init__(self) -> None:
        pass
    
    def masks(self, N) -> list[IPMask]: 
        return [
            IPMask.gaussian(N),
            IPMask.bimodal(N),
            IPMask.trimodal(N, linear_rate = 1/3),
            IPMask.trimodal(N, linear_rate = 2/3),
            IPMask.quadrimodal(N)
        ]
    
class GaussianSpace(IpTargetDistributionSpace): 
    def __init__(self, std_range = [0.15 ,0.25, 0.314]) -> None:
        self.std_range = std_range 
    
    def masks(self, N): 
        return [IPMask.gaussian(N, std) for std in self.std_range]

class Bimodal(IpTargetDistributionSpace): 
    def __init__(self, mu_range = [0.72, 0.92], std_range = [0.07, 0.14, 0.21]) -> None:
        self.mu_range = mu_range
        self.std_range = std_range

    def masks(self, N): 
        return [IPMask.gaussian(N, std) for std in itertools.product(*[self.mu_range, self.std_range])]

class TrimodalSpace(IpTargetDistributionSpace): 
    def __init__(self, mu_range = [0.72, 0.92], std_bim = [0.07,  0.21], std_lin_range = [0.25, 0.15], lin_rate_space = [1/3, 2/3]) -> None:
        pass

class QuadrimodalSpace(IpTargetDistributionSpace): 
    def __init__(self, eta_range, epochs_range, stds) -> None:
        pass

    
class IPHyperparametersSpace(): 
    def __init__(self, eta_range=[0.000025], epochs_range=[0], dist_space: IpTargetDistributionSpace = IpTargetDistributionSpace()) -> None:
        self.eta_range = eta_range
        self.epoch_range = epochs_range
        self.dist_space = dist_space


    def get_configs(self, inital_reservoir_configs: list[ReservoirConfiguration] = [ReservoirConfiguration()], ) -> list[IPReservoirConfiguration]:
        configs: list[IPReservoirConfiguration] = []
        for eta, epochs in itertools.product(*[self.eta_range, self.epoch_range]):
            for initial_config in inital_reservoir_configs:
                for mask in self.dist_space.masks(initial_config.N):
                    configs.append(IPReservoirConfiguration(config=initial_config, mask=mask, eta=eta, epochs=epochs))

        return configs
    
class GrdiSearchSpace(): 
    def __init__(self, vanillaReservoirSpace: ReservoirHyperparamSpace, ipHyperparamSpace = IPHyperparametersSpace(), lambda_thikonov_range=[0, 0.05, 0.1, 0.2]) -> None:
        self.vavanillaReservoirSpace = vanillaReservoirSpace
        self.ipHyperparamSpace = ipHyperparamSpace

        self.lambda_thikonov_range = lambda_thikonov_range

        self.vanilla_configs = vanillaReservoirSpace.get_configs()
        self.ip_configs = ipHyperparamSpace.get_configs(self.vanilla_configs)

    def get_configs(self, evaluate_vanilla = True): 
        return self.ip_configs if evaluate_vanilla == False else self.ip_configs + self.vanilla_configs
    
    def evaluate(self, data: TimeseriesDATA = MG17(), metric: Metric = MSE(), repetitions=10 ,transient = 100, evaluate_vanilla=True, grid_search_name = "GridSearch",):
        results = [["Model", "Lambda", f"Mean {metric.name}", f"Standard Deviation", "Description"]]

        X_TR, Y_TR = data.TR()
        X_VAL, Y_VAL = data.VAL()
        #X_TS, Y_TS = data.TS()

        for mi, config in enumerate(self.get_configs(evaluate_vanilla=evaluate_vanilla)):

            values = np.zeros([repetitions, len( self.lambda_thikonov_range)])

            for ri in range(repetitions):

                model = config.build_up_model(X_TR)
    
                for li, lambda_thikonov in enumerate( self.lambda_thikonov_range):
                    esn = EchoStateNetwork(model)
                    esn.train(X_TR, Y=Y_TR, transient=transient, lambda_thikonov=lambda_thikonov)
                    y_pred = esn.predict(X_VAL)

                    values[ri,li] = metric.evaluate(y_pred, Y_VAL)
            
            means = np.mean(values, axis=0)
            stds = np.std(values, axis = 0)

            for li, lambda_thikonov in enumerate( self.lambda_thikonov_range):
                config.set_lambda(lambda_thikonov) # Just for plotting reasons.
                results.append([f"Model {mi}", lambda_thikonov, means[li], stds[li], config.description()])

        df = pd.DataFrame(results[1:None], columns=[results[0]])
        df.sort_values(df.columns[2])

        if not os.path.exists("./GRID-SEARCHES/"):
            os.mkdir("./GRID-SEARCHES/")

        super().__init__()

        df.to_csv(f"{grid_search_name}.csv")
        return results

            


            
        #data=data, repetitions=repetitions, transient=transient, estrinsic_metrics=[metric], intrinsic_metrics=other_intrinsics, lambadas=self.lambda_ridge_range)