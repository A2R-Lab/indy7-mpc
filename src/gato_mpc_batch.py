import numpy as np
import pinocchio as pin
from utils import rk4
import torch
import bindings.batch_sqp as batch_sqp
import time

class GATO_Batch:
    def __init__(self, N=32, dt=0.01, batch_size=4, stats=None):
        self.N = N
        self.dt = dt
        self.solver = batch_sqp.SQPSolverfloat_4()
        
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
        
    def solve(self, xcur_batch, eepos_goals_batch, XU_batch):
        # xcur_batch: (batch_size, state_size)
        # eepos_goals_batch: (batch_size, N * 3)
        # XU_batch: (batch_size, N*(state_size+control_size)-control_size)
        
        result = self.solver.solve(XU_batch, self.dt, xcur_batch, eepos_goals_batch)
        
        XU_batch = result["xu_trajectory"]
        self.stats['solve_time']['values'].append(result["solve_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iterations"])
        for i in range(len(result["pcg_stats"])):
            self.stats['pcg_iters']['values'].append(result["pcg_stats"][i]["pcg_iterations"])
        for i in range(len(result["line_search_stats"])):
            self.stats["step_size"]["values"].append(result["line_search_stats"][i]["step_size"])
        return XU_batch
        
    def reset(self):
        self.solver.reset()
        
    def get_stats(self):
        return self.stats

class MPC_GATO_Batch:
    def __init__(self, model, N=32, dt=0.01, batch_size=4):
        self.model = model
        self.data = model.createData()
        #self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.array([0, 0, -9.81])
        
        self.solver = GATO_Batch(N, dt, batch_size)
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
        
        xcur_batch = np.tile(xstart, (self.solver.batch_size, 1))
        endpoint = endpoints[0]
        endpoint = np.hstack([endpoint, np.zeros(3)])
        eepos_goal = np.tile(endpoint, self.solver.N).T
        eepos_goal_batch = np.tile(eepos_goal, (self.solver.batch_size, 1))
        
        stats = {
            'solve_times': [],
            'goal_distances': [],
            'control_inputs': []
        }
        
        # Initialize trajectory
        XU_batch = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
        XU_batch[0:self.nx] = xcur[0, :]
        XU_batch = np.tile(XU_batch, (self.solver.batch_size, 1))
        XU_batch = self.solver.solve(xcur_batch, eepos_goal_batch, XU_batch)
        
    
        current_goal_idx = 0
        time_accumulated = 0.0 
        sim_dt = 0.01  # hard coded timestep for now, realistically should be trajopt solver time
        num_steps = int(sim_time / sim_dt)
        
        print(f"Running MPC with {num_steps} steps, sim_dt: {sim_dt}")
        print("Endpoints:")
        print(endpoints)
        
        for i in range(num_steps):            
            if i == num_steps - 1:
                print("Maximum steps reached without convergence")
                
            # Optimize trajectory
            self.solver.reset()
            #self.solver.solver.resetRho()
            #self.solver.solver.resetLambda()
            
            start_time = time.time()
            XU_batch_new = self.solver.solve(xcur_batch, eepos_goal_batch, XU_batch)
            end_time = time.time()
            
            # check if all trajectories are equal
            for j in range(self.solver.batch_size):
                trajectories_equal = np.allclose(
                    XU_batch_new[0], 
                    XU_batch_new[j],
                    rtol=1e-5,
                    atol=1e-5
                )
                if not trajectories_equal:
                    print(f"Trajectory {j} is not equal to trajectory 0!")
                    break
                
            xcur = xcur_batch[0, :]
            xcur = np.expand_dims(xcur, axis=0)
            eepos_goal = eepos_goal_batch[0, :]
            eepos_goal = np.expand_dims(eepos_goal, axis=0)
            XU = XU_batch[0, :]
            XU = np.expand_dims(XU, axis=0)
            xu_new = XU_batch_new[0, :]
            xu_new = np.expand_dims(xu_new, axis=0)
            
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
            if goaldist < 0.12 and current_goal_idx < len(endpoints) - 1:
                current_goal_idx += 1
                endpoint = endpoints[current_goal_idx]
                endpoint = np.hstack([endpoint, np.zeros(3)])
                eepos_goal = np.tile(endpoint, self.solver.N).T
                eepos_goal = np.expand_dims(eepos_goal, axis=0)
                eepos_goal_batch = np.tile(eepos_goal, (self.solver.batch_size, 1))
                print(f"Reached intermediate goal {current_goal_idx}, moving to next goal: {endpoint}")
            elif goaldist < 0.12 and current_goal_idx == len(endpoints) - 1:
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
            
            # update entire batch
            for j in range(self.solver.batch_size):
                xcur_batch[j, :] = xcur[0, :]
                XU_batch[j, :-(sim_steps)*(self.nx+self.nu) or len(XU_batch)] = XU[0, :-(sim_steps)*(self.nx+self.nu) or len(XU)]
                XU_batch[j, :self.nx] = XU[0, :self.nx]
                XU_batch[j, -self.nx:] = XU[0, -self.nx:]
            
            
        return self.xpath, stats
    
    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation