import numpy as np
import pinocchio as pin
from utils import rk4
import torch
import bindings.batch_sqp as batch_sqp
import time

class GATO_Batch_Sample:
    def __init__(self, N=32, dt=0.01, batch_size=4, stats=None, f_ext_std=1.0):
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
        if f_ext_std != 0.0:
            self.f_ext_batch = np.random.normal(0, f_ext_std, (self.batch_size, 6))
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[0] = np.array([0, 0, 0, 0, 0, 0])  
        else:
            self.f_ext_batch = np.zeros((self.batch_size, 6))
        #self.f_ext_last = np.array([0, 0, 0, 0, 0, 0])
        
        print("f_ext_batch:")
        print(self.f_ext_batch)
        
        self.solver.set_external_wrench_batch(self.f_ext_batch)
        
        
        
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

class MPC_GATO_Batch_Sample:
    def __init__(self, model, N=32, dt=0.01, batch_size=4, constant_f_ext=None, resample_f_ext=False, f_ext_std=1.0, f_ext_resample_std=2.0):
        
        self.model = model
        self.test_model = model
        self.data = model.createData()
        self.test_data = model.createData()
        #self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.array([0, 0, -9.81])
        
        self.solver = GATO_Batch_Sample(N, dt, batch_size, f_ext_std=f_ext_std)
        self.xpath = []  # Store trajectory for visualization
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = len(self.model.joints) - 1 # needed for some reason
        
        self.resample_f_ext = resample_f_ext
        self.f_ext_resample_std = f_ext_resample_std
        
        self.actual_f_ext = np.zeros((6))
        if constant_f_ext is not None:
            force = pin.Force(constant_f_ext[:3], np.array([0, 0, 0]))
            forces = pin.StdVec_Force()
            for _ in range(self.model.njoints-1):
                forces.append(pin.Force.Zero())
            forces.append(force)
            self.constant_f_ext = forces
        else:
            self.constant_f_ext = None
        
    def run_mpc(self, xstart, endpoints, sim_dt=0.001, sim_time=5):        
        xcur = xstart
        x_last = xstart
        best_id = 0
        u_last = np.zeros(self.nu)
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
        XU = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
        XU[0:self.nx] = xcur # need better initialization
        XU_batch = np.tile(XU, (self.solver.batch_size, 1))
        XU_batch = self.solver.solve(xcur_batch, eepos_goal_batch, XU_batch)
        XU_best = XU_batch[0, :]
        
        current_goal_idx = 0
        time_accumulated = 0.0 
        num_steps = int(sim_time / sim_dt)
        
        print(f"Running MPC with {num_steps} steps, sim_dt: {sim_dt}")
        print("Endpoints:")
        print(endpoints)
        
        for i in range(num_steps):
            if i == num_steps - 1:
                print("Maximum steps reached without convergence")
                
            # # set external wrench
            if np.any(self.constant_f_ext) is None:
                # Create empty force vector
                self.constant_f_ext = pin.StdVec_Force()
                # Add zero forces for all joints except the last one
                for _ in range(self.model.njoints-1):
                    self.constant_f_ext.append(pin.Force.Zero())
                
                # Convert force to the local frame of the end effector (joint 6)
                pin.forwardKinematics(self.model, self.data, xcur[:self.nq])
                pin.updateFramePlacements(self.model, self.data)
                
                # Create force in world frame
                world_force = pin.Force(self.actual_f_ext[:3], self.actual_f_ext[3:])
                
                # Transform force from world frame to joint's local frame
                local_force = self.data.oMi[6].actInv(world_force)
                
                # Add the transformed force to the last joint
                self.constant_f_ext.append(local_force)
                    
            x_last = xcur
            u_last = XU_best[self.nx:self.nx+self.nu]
            # Simulate forward
            sim_time_step = sim_dt  # hard coded timestep for now, realistically should be trajopt solver time
            sim_steps = 0  # full solver.dt steps completed
            q = xcur[:self.nq]
            v = xcur[self.nq:self.nx]
            while sim_time_step > 0:
                timestep = min(sim_time_step, self.solver.dt - time_accumulated)
                
                current_knot = int(time_accumulated / self.solver.dt)
                if current_knot >= self.solver.N:
                    current_knot = self.solver.N - 1
                
                u = XU_best[current_knot*(self.nx+self.nu)+self.nx:(current_knot+1)*(self.nx+self.nu)]
                q, v = rk4(self.model, self.data, q, v, u, timestep, self.constant_f_ext)
                
                time_accumulated += timestep
                sim_time_step -= timestep
                self.xpath.append(q)
                
                # if full solver.dt step is completed
                if abs(time_accumulated - self.solver.dt) < 1e-10:
                    sim_steps += 1
                    time_accumulated = 0.0
                    
            xcur = np.hstack([q, v])
            for j in range(self.solver.batch_size):
                xcur_batch[j, :] = xcur
                XU_batch[j, :self.nx] = xcur # first state
                XU_batch[j, self.nx:] = XU_best[self.nx:]
            # # shift trajectory
            # if sim_steps > 0:
            #     XU_batch[0, :-(sim_steps)*(self.nx+self.nu) or len(XU_batch)] = XU_best[(sim_steps)*(self.nx+self.nu):]
            # # update batch
            # for j in range(self.solver.batch_size):
            #     xcur_batch[j, :] = xcur
            #     XU_batch[j, :self.nx] = xcur # first state
            #     XU_batch[j, :-(sim_steps)*(self.nx+self.nu) or len(XU_batch)] = XU_batch[j, :-(sim_steps)*(self.nx+self.nu) or len(XU_batch)]

            cur_eepos = self.eepos(xcur[:self.nq])
            eepos_goal = eepos_goal_batch[0, :]
            goaldist = np.linalg.norm(cur_eepos - eepos_goal[:3])
         
            # check goal distance
            if goaldist < 0.095 and current_goal_idx < len(endpoints) - 1:
                current_goal_idx += 1
                endpoint = endpoints[current_goal_idx]
                endpoint = np.hstack([endpoint, np.zeros(3)])
                eepos_goal = np.tile(endpoint, self.solver.N).T
                eepos_goal_batch = np.tile(eepos_goal, (self.solver.batch_size, 1))
                print(f"Reached intermediate goal {current_goal_idx}, moving to next goal: {endpoint}")
            elif goaldist < 0.095 and current_goal_idx == len(endpoints) - 1:
                print(f"Reached final goal {current_goal_idx}")
                break


            # ----- Optimize trajectory -----
            
            self.solver.reset()
            self.solver.solver.resetRho()
            #self.solver.solver.resetLambda()
            
            start_time = time.time()
            XU_batch_new = self.solver.solve(xcur_batch, eepos_goal_batch, XU_batch)
            end_time = time.time()
            
            # find best traj after simulation step (most closely matched dynamics)
            best_id = self.find_best_trajectory_id(x_last, u_last, xcur, sim_dt)
            XU_best = XU_batch_new[best_id, :]
            # -----
            
            # print every 0.05 seconds
            if abs((i * sim_dt) % 0.05) < 1e-5:
                print(f"t: {i * sim_dt:8.3f}, goal dist: {goaldist:8.5f}, best_id: {best_id}")
                # Log statistics
                stats['goal_distances'].append(float(round(goaldist, 5)))
                control = XU_best[self.nx:self.nx+self.nu]
                stats['control_inputs'].append(np.round(control.copy(), 5))
                stats['solve_times'].append(float(round(end_time - start_time, 5)))
                
            new_force = np.random.normal(self.constant_f_ext[6].linear, 0.1)
            force = pin.Force(new_force[:3], np.array([0, 0, 0]))
            forces = pin.StdVec_Force()
            for _ in range(self.model.njoints-1):
                forces.append(pin.Force.Zero())
            forces.append(force)
            self.constant_f_ext = forces
            
        return self.xpath, stats
    
    def find_best_trajectory_id(self, x_last, u_last, xcur, dt):
        best_error = np.inf
        best_id = None
        f_ext_best = np.zeros(6)
        
        
        for i in range(self.solver.batch_size):
            model = self.model.copy()
            data = self.data.copy()
            f_ext = self.solver.f_ext_batch[i, :]
            
            # Create empty force vector
            forces = pin.StdVec_Force()
            # Add zero forces for all joints except the last one
            for _ in range(self.model.njoints-1):
                forces.append(pin.Force.Zero())
            
            # Update kinematics to get current transformations
            pin.forwardKinematics(model, data, x_last[:self.nq])
            pin.updateFramePlacements(model, data)
            
            # Create force in world frame
            world_force = pin.Force(f_ext[:3], f_ext[3:])
            
            # Transform force from world frame to joint's local frame
            local_force = data.oMi[6].actInv(world_force)
            
            # Add the transformed force to the last joint
            forces.append(local_force)
            
            q, v = rk4(model, data, x_last[:self.nq], x_last[self.nq:self.nx], u_last, dt, forces)
            x_next = np.hstack([q, v])
            error = np.linalg.norm(x_next - xcur)
            if error < best_error:
                best_error = error
                best_id = i
                f_ext_best = f_ext
        if self.resample_f_ext:
            for i in range(self.solver.batch_size):
                self.solver.f_ext_batch[i, :] = f_ext_best
            self.solver.f_ext_batch = f_ext_best + np.random.normal(0, self.f_ext_resample_std, self.solver.f_ext_batch.shape)
            for i in range(self.solver.batch_size):
                self.solver.f_ext_batch[i, 3:] = 0.0
            self.solver.f_ext_batch[0] = np.array([0, 0, 0, 0, 0, 0])
            self.solver.solver.set_external_wrench_batch(self.solver.f_ext_batch)
                
        return best_id
            
    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation