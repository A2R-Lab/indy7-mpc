import numpy as np
import pinocchio as pin
from utils import rk4

class MPC_OSQP:
    def __init__(self, model, sqp_optimizer, solver):
        self.model = model
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.sqp_optimizer = sqp_optimizer
        self.solver = solver
        self.xpath = []  # Store trajectory for visualization
        
    def run_mpc(self, xstart, endpoints, num_steps=500): # (nx, (3, ), num_steps)
        nq = self.solver.nq
        nv = self.solver.nv
        nx = self.solver.nx
        nu = self.solver.nu
        
        xcur = xstart
        endpoint_ind = 0
        endpoint = endpoints[endpoint_ind]
        eepos_goal = np.tile(endpoint, self.solver.N).T
        
        # Initialize trajectory
        XU = np.zeros(self.solver.N*(nx+nu)-nu)
        XU = self.sqp_optimizer.sqp(xcur, eepos_goal, XU)
        
        for i in range(num_steps):
            # Check if we need to switch goals
            cur_eepos = self.solver.eepos(xcur[:nq])
            goaldist = np.linalg.norm(cur_eepos - eepos_goal[:3])
            
            if goaldist < 1e-1:
                print('switching goals')
                endpoint_ind = (endpoint_ind + 1) % len(endpoints)
                endpoint = endpoints[endpoint_ind]
                eepos_goal = np.tile(endpoint, self.solver.N).T
            
            print(goaldist)
            if goaldist > 1.1:
                print("breaking on big goal dist")
                break
                
            # Optimize trajectory
            xu_new = self.sqp_optimizer.sqp(xcur, eepos_goal, XU)
            
            # Simulate forward using control
            trajopt_time = 0.01  # hard coded timestep
            sim_time = trajopt_time
            sim_steps = 0  # full steps taken
            
            while sim_time > 0:
                timestep = min(sim_time, self.solver.dt)
                control = XU[sim_steps*(nx+nu)+nx:(sim_steps+1)*(nx+nu)]
                xcur = np.vstack(rk4(self.model, self.solver.data, xcur[:nq], xcur[nq:nx], control, timestep)).reshape(-1)
                
                if timestep > 0.5 * self.solver.dt:
                    sim_steps += 1
                
                sim_time -= timestep
                self.xpath.append(xcur[:nq])
                
            # Update trajectory with new solution
            if sim_steps > 0:
                XU[:-(sim_steps)*(nx+nu) or len(XU)] = xu_new[(sim_steps)*(nx+nu):]
                
            # Update first and last states
            XU[:nx] = xcur.reshape(-1)  # first state is current state
            XU[-nx:] = np.hstack([np.ones(nq), np.zeros(nv)])  # last state is target
            
        return self.xpath