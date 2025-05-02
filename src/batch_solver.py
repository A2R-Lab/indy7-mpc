import numpy as np
import torch
import sys 
import os

sys.path.append('../../')
import bindings.batch_sqp as batch_sqp

class GatoBatchSolver:
    def __init__(self, config_path, f_ext_std=1.0, f_ext_resample_std=0.0):
        self.config = yaml.load(open(config_path))
        self.solver = batch_sqp.SQPSolverfloat()
        self.N = self.config['solver']['N_horizon']
        self.dt = self.config['solver']['dt']
        self.batch_size = self.config['solver']['N_batch']
        
        self.stats = {
            'solve_time': {'values': [], 'unit': 'us', 'multiplier': 1},
            'pcg_iters': {'values': [], 'unit': '', 'multiplier': 1},
            "step_size": {"values": [], "unit": "", "multiplier": 1},
            'sqp_iters': {'values': [], 'unit': '', 'multiplier': 1}
        }

        if f_ext_std != 0.0:
            self.f_ext_std = f_ext_std
            self.f_ext_batch = np.random.normal(0, f_ext_std, (self.batch_size, 6))
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        else:
            self.f_ext_batch = np.zeros((self.batch_size, 6))

        self.resample_f_ext = True if f_ext_resample_std > 0.0 else False
        self.f_ext_resample_std = f_ext_resample_std
        
        print("f_ext_batch:")
        print(np.array2string(self.f_ext_batch, precision=4, suppress_small=True))
        self.solver.set_external_wrench_batch(self.f_ext_batch)

        
    def solve(self, XU_batch, x_0_batch, ref_traj_batch):
        
        result = self.solver.solve(XU_batch, x_0_batch, ref_traj_batch)
        self.stats['solve_time']['values'].append(result["solve_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iterations"])
        for i in range(len(result["pcg_stats"])):
            self.stats['pcg_iters']['values'].append(result["pcg_stats"][i]["pcg_iterations"])
        for i in range(len(result["line_search_stats"])):
            self.stats["step_size"]["values"].append(result["line_search_stats"][i]["step_size"])
            
        return result["xu_trajectory"], result["solve_time_us"]
    
    def sim_forward(self, x, u, sim_dt):
        x_next_batch = self.solver.sim_forward(x, u, sim_dt)
        return x_next_batch
    
    def find_best_idx(self, x_last, u_last, xs, dt):
        best_error, best_idx = np.inf, None
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        for i in range(self.batch_size):
            error = np.linalg.norm(x_next_batch[i, :] - xs)
            if error <= best_error:
                best_error, best_idx = error, i
                
        return best_idx
    
    def resample_f_ext_batch(self, best_idx):
        if self.resample_f_ext:
            f_ext_best = self.f_ext_batch[best_idx, :]
            self.f_ext_batch[:, :] = f_ext_best
            self.f_ext_batch += np.random.normal(0, self.f_ext_resample_std, self.f_ext_batch.shape)
            self.f_ext_batch[best_idx, :] = f_ext_best
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.f_ext_batch *= 0.98
            self.solver.set_external_wrench_batch(self.f_ext_batch)

    def reset(self):
        self.solver.reset()
        
    def reset_lambda(self):
        self.solver.reset_lambda()
    
    def reset_rho(self):
        self.solver.reset_rho()
        
    def get_stats(self):
        return self.stats