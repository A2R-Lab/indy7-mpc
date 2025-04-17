import numpy as np
import torch
import bindings.batch_sqp as batch_sqp

class GATO_Batch_Solver:
    def __init__(self, N=32, dt=0.01, batch_size=4, stats=None, f_ext_std=1.0):
        
        self.N = N
        self.dt = dt
        
        solver_map = {
            1: batch_sqp.SQPSolverfloat_1,
            2: batch_sqp.SQPSolverfloat_2,
            4: batch_sqp.SQPSolverfloat_4,
            8: batch_sqp.SQPSolverfloat_8,
            16: batch_sqp.SQPSolverfloat_16,
            32: batch_sqp.SQPSolverfloat_32,
            64: batch_sqp.SQPSolverfloat_64,
            128: batch_sqp.SQPSolverfloat_128,
            256: batch_sqp.SQPSolverfloat_256
        }
        if batch_size not in solver_map:
            raise ValueError(f"Batch size {batch_size} not supported")
        
        self.batch_size = batch_size
        self.solver = solver_map[batch_size]()
        
        self.stats = stats or {
            'solve_time': {'values': [], 'unit': 'us', 'multiplier': 1},
            'sqp_iters': {'values': [], 'unit': '', 'multiplier': 1},
            'pcg_iters': {'values': [], 'unit': '', 'multiplier': 1},
            "step_size": {"values": [], "unit": "", "multiplier": 1}
        }
        
        if f_ext_std != 0.0:
            self.f_ext_batch = np.random.normal(0, f_ext_std, (self.batch_size, 6))
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[0] = np.array([0, 0, 0, 0, 0, 0])  
        else:
            self.f_ext_batch = np.zeros((self.batch_size, 6))
        #self.f_ext_last = np.array([0, 0, 0, 0, 0, 0])
        
        print("f_ext_batch:")
        print(self.f_ext_batch)
        
        self.set_external_wrench_batch(self.f_ext_batch)
        
    # xs_batch: (batch_size, state_size)
    # eepos_goals_batch: (batch_size, N * 6)
    # XU_batch: (batch_size, N*(state_size+control_size)-control_size)
    def solve(self, xs_batch, eepos_goals_batch, XU_batch):
        
        result = self.solver.solve(XU_batch, self.dt, xs_batch, eepos_goals_batch)
        
        self.stats['solve_time']['values'].append(result["solve_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iterations"])
        for i in range(len(result["pcg_stats"])):
            self.stats['pcg_iters']['values'].append(result["pcg_stats"][i]["pcg_iterations"])
        for i in range(len(result["line_search_stats"])):
            self.stats["step_size"]["values"].append(result["line_search_stats"][i]["step_size"])
        return result["xu_trajectory"], result["solve_time_us"]
    
    def find_best_idx(self, x_curr, x_last, u_last, dt):
        best_idx = 0
        best_error = np.inf
        
        x_next_batch = self.sim_forward(x_curr, u_last, dt)
        
        for i in range(self.batch_size):
            error = np.linalg.norm(x_next_batch[i] - x_curr)
            if error < best_error:
                best_error = error
                best_idx = i
                
        # TODO: resample f_ext_batch
        
        return best_idx
        
    def reset(self):
        self.solver.reset()
        
    def reset_rho(self):
        self.solver.resetRho()
        
    def reset_lambda(self):
        self.solver.resetLambda()
        
    def set_external_wrench_batch(self, f_ext_batch):
        self.solver.set_external_wrench_batch(f_ext_batch)
        
    def sim_forward(self, x, u, dt):
        # returns x_next_batch (batch_size, state_size)
        return self.solver.sim_forward(x, u, dt)
    
    def get_stats(self):
        return self.stats