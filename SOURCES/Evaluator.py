from Configurations import ReservoirConfiguration
from DATA import TimeseriesDATA
from Metrics import *

import pandas as pd
import numpy as np
import pickle
import os

"""
  Class generalizing the evaluation procedure for all metrics implemented up to now, provdinging 
  unified methods for measurment, comparison and optimal parameter research. 

"""
class Evaluator(): 
    def __init__(self, save_results_csv = True, save_models_pickle = True, path = "./../EXPERIMENTS/", experiment_name = "Experiment0"): 
        self.save_results_csv = save_results_csv
        self.save_models_pickle = save_models_pickle
        self.name = experiment_name
    
        if save_results_csv or save_models_pickle:
            self.outdir = f"{path}"

            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
        super().__init__()

    @staticmethod
    def evaluate_estrinsic(model: Reservoir, data: TimeseriesDATA, metric: Metric, transient = 100, lambda_thikonov = 0): 
        X_TR, Y_TR = data.TR()
        #X_VAL, Y_VAL = data.VAL() Validation set is expectred to be empty now.
        X_TS, Y_TS = data.TS()

        esn = EchoStateNetwork(model)
        esn.train(X_TR, Y=Y_TR, transient=transient, lambda_thikonov=lambda_thikonov)
        y_pred = esn.predict(X_TS)

        return metric.evaluate(y_pred, Y_TS)

        
    def evaluate_multiple(self, model_configs: list[ReservoirConfiguration] , data: TimeseriesDATA, repetitions: int, transient: int = 100, 
                         estrinsic_metrics: list[EstrinsicMetric] = [MSE(), NRMSE()], intrinsic_metrics: list[IntrinsicMetric] = []):


        X_TR, Y_TR = data.TR()
        #X_VAL, Y_VAL = data.VAL() Validation set is expectred to be empty now.
        X_TS, Y_TS = data.TS()

                
        if intrinsic_metrics == []: 
            intrinsic_metrics = [Rho(),MLLE(X_TS),  DeltaPhi(), MC(), Neff() ]

        len_est = len(estrinsic_metrics)
        len_int = len(intrinsic_metrics)
        metrics_num = len_est + len_int 

        results_matrix = np.zeros([2*len(model_configs), 1 + metrics_num])

        for model_index, model_config in enumerate(model_configs):
            config_results_matrix = np.zeros([repetitions, metrics_num])

            for i in range(repetitions):
                model = model_config.build_up_model(U_TR=X_TR, transient=transient)

                esn = EchoStateNetwork(model)
                esn.train(X_TR, Y=Y_TR, transient=transient, lambda_thikonov=model_config.lambda_thikonv)
                y_pred = esn.predict(X_TS)

                for j, e_metric in enumerate(estrinsic_metrics):
                    config_results_matrix[i,j] = e_metric.evaluate(y_pred, Y_TS)

                for j, i_metric in enumerate(intrinsic_metrics): 
                    config_results_matrix[i, len_est + j] = i_metric.evaluate(model)

            
            np_config_results = np.array(config_results_matrix)

            results_matrix[model_index*2, 0] =  model_index #f"Mean - {model_config.name}"
            results_matrix[model_index*2, 1:None] = np.mean(np_config_results, axis = 0)

            results_matrix[model_index*2 +1, 0] = model_index #f"Std - {model_config.name}"
            results_matrix[model_index*2 + 1, 1:None] = np.std(np_config_results, axis = 0)

        model_names = list(map(lambda m:m.name, model_configs))
        column_names = list(map(lambda m:m.name, np.concatenate([estrinsic_metrics, intrinsic_metrics]).ravel()))
        column_names.insert(0, "Model Index")
       
        df = pd.DataFrame(results_matrix, columns=column_names)

        df.insert(1, "Aggregation", ["Mean" if i % 2 == 0 else "Std" for i in range(2*len(model_configs))])
        df.insert(0, "Model Name", [model_names[int(i/2)] for i in range(2*len(model_names))])

        if self.save_models_pickle:
            self.save_models(model_configs)    

        if self.save_results_csv: 
            df.to_csv(f"{self.outdir}/RESULTS-{self.name}.CSV")

        return df 
       
    """
        Saves a list of model in a unique pickle file. 
        Created to automatize comparison procedures
    """
    def save_models(self, model_config: list[ReservoirConfiguration]): 
        with open(f"{self.outdir}/MODELS.pickle", "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump(model_config, outfile)

    """
        Load multiple model from a unique pickle file. 
        Created to repeat already executed or previously defined experiments.
    """
    def load_models(self) -> list[ReservoirConfiguration]: 
        with open(f"{self.outdir}/MODELS.pickle", "rb") as infile:
            return pickle.load(infile) 


    """
        Saves the configuration of a single model. 
        Created to store best model found with grid searches.
    """
    def save_model_config(self, model_config: ReservoirConfiguration, prefix=""):
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        
        if not os.path.exists(f"{self.outdir}/MODELS"):
             os.mkdir(f"{self.outdir}/MODELS")
          
        with open(f"{self.outdir}/MODELS/{prefix}{model_config.name}.pickle", "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump(model_config, outfile)
    

    """
        Load the configuration of a single model. 
        Created to load the best models found with grid searches.
    """
    def load_model_config(self, filename:str) -> ReservoirConfiguration:
        with open(f"{self.outdir}/MODELS/{filename}.pickle", "rb") as infile:
            return pickle.load(infile) 
        

