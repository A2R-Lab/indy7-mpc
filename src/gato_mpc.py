import numpy as np
import pinocchio as pin
from utils import rk4
import torch
import bindings.batch_sqp as batch_sqp
import time

class GATO:
    def __init__(self, N=32, dt=0.01, stats=None):
        self.N = N
        self.dt = dt
        
        self.solver = batch_sqp.SQPSolverfloat_1()
        self.stats = stats or {
            'solve_time': {'values': [], 'unit': 'us', 'multiplier': 1},
            'sqp_iters': {'values': [], 'unit': '', 'multiplier': 1},
            'pcg_iters': {'values': [], 'unit': '', 'multiplier': 1},
            "step_size": {"values": [], "unit": "", "multiplier": 1}
        }

    def solve(self, xcur, eepos_goals, XU):
        result = self.solver.solve(XU, self.dt, xcur, eepos_goals)
        
        XU = result["xu_trajectory"]
        self.stats['solve_time']['values'].append(result["solve_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iterations"])
        for i in range(len(result["pcg_stats"])):
            self.stats['pcg_iters']['values'].append(result["pcg_stats"][i]["pcg_iterations"])
        for i in range(len(result["line_search_stats"])):
            self.stats["step_size"]["values"].append(result["line_search_stats"][i]["step_size"])
        return XU
    
    def reset(self):
        self.solver.reset()
    
    def get_stats(self):
        return self.stats


class MPC_GATO:
    def __init__(self, model, N=32, dt=0.01):
        self.model = model
        self.data = model.createData()
        self.solver = GATO(N, dt)
        self.xpath = []  # Store trajectory for visualization
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = len(self.model.joints) - 1 # needed for some reason
        
        
    def run_mpc(self, xstart, endpoints, sim_time=5):        
        xcur = xstart
        xcur = np.expand_dims(xcur, axis=0)
        endpoint = endpoints[0]
        endpoint = np.hstack([endpoint, np.zeros(3)])
        eepos_goal = np.tile(endpoint, self.solver.N).T
        eepos_goal = np.expand_dims(eepos_goal, axis=0)
        
        stats = {
            'solve_times': [],
            'goal_distances': [],
            'control_inputs': []
        }
        
        # Initialize trajectory
        XU = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
        XU[0:self.nx] = xcur[0, :]
        XU = np.expand_dims(XU, axis=0)
        XU = self.solver.solve(xcur, eepos_goal, XU)
        
    
        current_goal_idx = 0
        time_accumulated = 0.0 
        sim_dt = 0.02  # hard coded timestep for now, realistically should be trajopt solver time
        num_steps = int(sim_time / sim_dt)
        
        print(f"Running MPC with {num_steps} steps, dt: {sim_dt}")
        print("Endpoints:")
        print(endpoints)
        
        for i in range(num_steps):            
            if i == num_steps - 1:
                print("Maximum steps reached without convergence")
                
            # Optimize trajectory
            self.solver.reset()
            
            start_time = time.time()
            xu_new = self.solver.solve(xcur, eepos_goal, XU)
            end_time = time.time()
            
            cur_eepos = self.eepos(xcur[0, :self.nq])
            goaldist = np.linalg.norm(cur_eepos - eepos_goal[0, :3])
            # Check if current time is close to a multiple of 0.05 within floating point precision
            if abs((i * sim_dt) % 0.05) < 1e-10:
                print(f"t: {i * sim_dt:8.3f}, goal dist: {goaldist:8.5f}")
                # Log statistics
                stats['goal_distances'].append(float(round(goaldist, 5)))
                control = XU[0, self.nx:self.nx+self.nu]
                stats['control_inputs'].append(np.round(control.copy(), 5))
                stats['solve_times'].append(float(round(end_time - start_time, 5)))
            
            # Check if current goal is reached and update to next goal
            if goaldist < 1e-2 and current_goal_idx < len(endpoints) - 1:
                current_goal_idx += 1
                endpoint = endpoints[current_goal_idx]
                endpoint = np.hstack([endpoint, np.zeros(3)])
                eepos_goal = np.tile(endpoint, self.solver.N).T
                eepos_goal = np.expand_dims(eepos_goal, axis=0)
                print(f"Reached intermediate goal {current_goal_idx}, moving to next goal")
            elif goaldist < 1e-2:
                print(f"Reached final goal {current_goal_idx}")
                break
            
            # Simulate forward
            sim_time_step = sim_dt  # hard coded timestep for now, realistically should be trajopt solver time
            sim_steps = 0  # full solver.dt steps completed
            
            while sim_time_step > 0:
                timestep = min(sim_time_step, self.solver.dt - time_accumulated)
                
                # Calculate which control interval we're in and get corresponding control
                current_interval = int(time_accumulated / self.solver.dt)
                control = XU[0, current_interval*(self.nx+self.nu)+self.nx:(current_interval+1)*(self.nx+self.nu)]
                
                # integrate forward to get next state
                q_next, v_next = rk4(self.model, self.data, xcur[0, :self.nq], xcur[0, self.nq:self.nx], control, timestep)
                xcur[0, :self.nq] = q_next
                xcur[0, self.nq:self.nx] = v_next
                
                time_accumulated += timestep
                sim_time_step -= timestep
                self.xpath.append(xcur[0, :self.nq].flatten())
                
                # Check if we've completed a full solver.dt step
                if abs(time_accumulated - self.solver.dt) < 1e-10:
                    sim_steps += 1
                    time_accumulated = 0.0
            
            # Update trajectory with new solution
            if sim_steps > 0:
                XU[0, :-(sim_steps)*(self.nx+self.nu) or len(XU)] = xu_new[0, (sim_steps)*(self.nx+self.nu):]
                
            # Update first and last states
            XU[0, :self.nx] = xcur[0, :]  # first state is current state
            XU[0, -self.nx:] = np.hstack([np.ones(self.nq), np.zeros(self.nv)])  # last state
            
        return self.xpath, stats
    
    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation