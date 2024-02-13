from Reservoir import ReservoirConfiguration
from DATA import TimeseriesDATA
from Metrics import *

import pandas as pd
import numpy as np

"""
  Class generalizing the evaluation procedure for all metrics implemented up to now, provdinging 
  unified methods for measurment, comparison and optimal parameter research. 

"""
class Evaluator(): 
    def __init__(self, save_csv = False, path = "./../RESULTS/.", experiment_name = "Experiment0"): 
         super().__init__()

    def evaluate_multiple(self,  model_configs: list[ReservoirConfiguration], data: TimeseriesDATA, repetitions: int, transient: int = 100, 
                         estrinsic_metrics: list[EstrinsicMetric] = [MSE(), NRMSE()], intrinsic_metrics: list[IntrinsicMetric] = [MC(), MLLE(), DeltaPhi(), Neff()]):
        
        X_TR, Y_TR = data.TR()
        #X_VAL, Y_VAL = data.VAL() Validation set is expectred to be empty now.
        X_TS, Y_TS = data.TS()

        len_est = len(estrinsic_metrics)
        len_int = len(intrinsic_metrics)
        metrics_num = len_est + len_int 

        results_matrix = np.zeros([2*len(model_configs), 1 + metrics_num])

        for model_index, model_config in enumerate(model_configs):
            config_results_matrix = np.zeros([repetitions, metrics_num])

            for i in range(repetitions):
                model = model_config.build_up_model()

                esn = EchoStateNetwork(model)
                esn.train(X_TR, Y=Y_TR, transient=transient)
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

        return df 
    
    def grid_search() -> ReservoirConfiguration: 
        return
    