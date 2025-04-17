import numpy as np
import torch
import sys 
import os
sys.path.append('../../') # to get bindings
sys.path.append('./src') # to get utils
import bindings.batch_sqp as batch_sqp
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Wrench
from rclpy.logging import get_logger

JOINT_STATE_TIMEOUT = 10.0

def figure8(x_amplitude=0.4, z_amplitude=0.4, offset=[0, 0.55, 0.35], period=5, dt=0.01, cycles=5):
    x = lambda t: offset[0] + x_amplitude * np.sin(t)
    y = lambda t: offset[1]
    z = lambda t: offset[2] + z_amplitude * np.sin(2*t)/2 + z_amplitude/2
    
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    points = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    
    return np.tile(points, cycles)

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

class GATO_Controller(Node):
    def __init__(self, ee_pos_reference_trajectory=None):
        super().__init__('gato_controller')
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            2)
        
        self.joint_commands_publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            2)
        
        self.external_force_publisher = self.create_publisher(
            Wrench, 
            'external_force', 
            2)
        
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        self.nu = 6
        self.nx = 12
        
        self.batch_size = 1
        self.N = 32
        self.dt = 0.01
        self.f_ext_std = 0.0
        
        self.solver = GATO_Batch_Solver(self.N, self.dt, self.batch_size, f_ext_std=self.f_ext_std)
        
        self.ee_pos_ref_traj = ee_pos_reference_trajectory
        self.ee_pos_ref_offset = 0.0
        
        self.ee_pos_traj_batch = np.tile(self.ee_pos_ref_traj[0:6*self.N], (self.solver.batch_size, 1))
        self.xs_batch = np.zeros((self.solver.batch_size, self.nx))
        self.XU_batch = np.zeros((self.solver.batch_size, self.solver.N*(18)-6))
        self.XU_best = np.zeros(self.solver.N*(18)-6)
        self.x_last = None
        self.u_last = None
        
        # stats
        self.tracking_errors = []
        self.solve_times = []
        self.ee_positions = []
        self.joint_positions = []
        
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        self.last_time = time.time()
        
    def joint_callback(self, msg):
        
        xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        RCLCPP_INFO(self.get_logger(), "xs: %s", xs)
        
        ee_pos = np.array([msg.effort[0], msg.effort[1], msg.effort[2]])
        
        if self.x_last is None:
            self.x_last = xs
            self.u_last = np.zeros(self.nu)
            return
        
        elapsed_time = time.time() - self.last_time
        self.last_time = time.time()
        self.ee_pos_ref_offset += elapsed_time / self.dt
        offset = int(self.ee_pos_ref_offset)
        self.ee_pos_traj_batch[:,0:6*self.N] = self.ee_pos_ref_traj[offset:offset+6*self.N]
        
        self.xs_batch[:,:self.nx] = xs
        self.XU_best[:self.nx] = xs
        self.XU_batch[:,:] = self.XU_best
        
        # ----- solve batch trajopt -----
        
        #start_time = time.time()
        self.solver.reset()
        self.solver.solver.resetRho()
        #self.solver.solver.resetLambda()
        XU_batch, solve_time_us = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, self.XU_batch)
        #end_time = time.time()
        
        best_idx = self.solver.find_best_idx(xs, self.x_last, self.u_last, elapsed_time)
        self.XU_best = XU_batch[best_idx]
        
        # ----- publish control -----
        
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * 6
        self.ctrl_msg.velocity = [0.0] * 6
        u = self.XU_best[self.nx:(self.nx+self.nu)]
        self.ctrl_msg.effort = [float(e) for e in u]
        self.joint_commands_publisher.publish(self.ctrl_msg)
        
        self.x_last, self.u_last = xs, u
        
        # ----- update stats -----
        self.tracking_errors.append(np.linalg.norm(ee_pos - self.ee_pos_traj_batch[0,:3]))
        self.solve_times.append(solve_time_us)
        self.ee_positions.append(ee_pos)
        self.joint_positions.append(xs[:6])
        
        print("\n--------------------------------\n")

    def check_timeout(self):
        current_time = time.time()
        if current_time - self.last_time > JOINT_STATE_TIMEOUT:
            self.get_logger().error(f'No joint state messages received for {JOINT_STATE_TIMEOUT} seconds. Exiting...')
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
            
def main(args=None):
    try:
        rclpy.init(args=args)
        reference_trajectory = figure8(x_amplitude=0.4, z_amplitude=0.4, offset=[0, 0.55, 0.35], period=5, dt=0.01, cycles=5)
        gato_controller = GATO_Controller(reference_trajectory)
        rclpy.spin(gato_controller)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_controller.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()