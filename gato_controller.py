import numpy as np
import torch
import sys 
import os
sys.path.append('./src')
sys.path.append('../../') # to get bindings
import bindings.batch_sqp as batch_sqp
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

JOINT_STATE_TIMEOUT = 5.0

def figure8():
    xamplitude = 0.4 # X goes from -xamplitude to xamplitude
    zamplitude = 0.8 # Z goes from -zamplitude/2 to zamplitude/2
    yoffset = 0.30
    period = 5 # seconds
    dt = 0.01 # seconds
    x = lambda t: xamplitude * np.sin(t)
    y = lambda t: 0.3
    z = lambda t: 0.1 + zamplitude * np.sin(2*t)/2 + zamplitude/2
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    points = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    return points


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
            
        step_sizes = []
        for i in range(len(result["line_search_stats"])):
            step_sizes.append(result["line_search_stats"][i]["step_size"])
        self.stats["step_size"]["values"].append(step_sizes)
        return XU_batch, step_sizes
        
    def reset(self):
        self.solver.reset()
        
    def get_stats(self):
        return self.stats






class GATO_Controller(Node):
    def __init__(self):
        super().__init__('gato_controller')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            1)
        self.publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            10)
        
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.jointstate_count = 0
        
        self.nq = 6
        self.nv = 6
        self.nu = 6
        self.nx = self.nq + self.nv
        
        self.batch_size = 1
        self.N = 32
        self.dt = 0.01
        
        self.solver = GATO_Batch_Sample(self.N, self.dt, self.batch_size, f_ext_std=0.0)
    
        self.fig8 = figure8()
        self.fig8_offset = 0
        
        self.eepos_g = np.tile(self.fig8[:6], (self.N, 1)).T
        #np.array(self.fig8[:self.fig8_offset+(6*self.N)], dtype=float)
        self.eepos_g_batch = np.tile(self.eepos_g, (self.batch_size, 1))
        
        self.XU_best = np.zeros(self.solver.N*(self.nx+self.nu)-self.nu)
        self.last_joint_state_time = time.time()
        # xs_batch = np.tile(np.zeros(self.nx), (self.solver.batch_size, 1))
        # XU_batch, step_sizes = self.solver.solve(xs_batch, self.eepos_g_batch, self.XU_best)
        # self.XU_best = XU_batch[0, :]
        
    def joint_callback(self, msg):
        print(f"dt: {time.time() - self.last_joint_state_time}")
        self.last_joint_state_time = time.time()
        self.jointstate_count += 1
        
        
        xs = np.hstack([np.array(msg.position, dtype=float), np.array(msg.velocity, dtype=float)])
        print(f"xs: {xs}")
        xs_batch = np.tile(xs, (self.solver.batch_size, 1))
        self.XU_best[0:self.nx] = xs
        XU_batch = np.tile(self.XU_best, (self.solver.batch_size, 1))
        
        self.solver.reset()
        #self.solver.solver.resetRho()
        #self.solver.solver.resetLambda()
        
        start_time = time.time()
        XU_batch_new, step_sizes = self.solver.solve(xs_batch, self.eepos_g_batch, XU_batch)
        end_time = time.time()
        print(f"batch sqp time: {1000 * (end_time - start_time)} ms")
        print(f"step sizes: {step_sizes}")
        
        best_tracker_idx = 0
        best_result = XU_batch_new[best_tracker_idx]
        u = best_result[self.nx:(self.nx+self.nu)]
        print(f"u: {u}")
        
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * self.nq # list(best_result[:self.nq])
        self.ctrl_msg.velocity = [0.0] * 6
        self.ctrl_msg.effort = [float(e) for e in u]
        self.publisher.publish(self.ctrl_msg)
        
        self.XU_best = XU_batch_new[best_tracker_idx]
        
        self.fig8_offset += 6
        self.fig8_offset %= len(self.fig8)
        copy1_len = min(self.fig8_offset + 6*self.N, len(self.fig8)) - self.fig8_offset
        copy2_len = 6*self.N - copy1_len
        self.eepos_g_batch[:,:copy1_len] = self.fig8[self.fig8_offset:self.fig8_offset+copy1_len]
        if copy2_len > 0:
            self.eepos_g_batch[:,-copy2_len:] = self.fig8[:copy2_len]
        
        print("--------------------------------\n\n")

    def check_timeout(self):
        current_time = time.time()
        if current_time - self.last_joint_state_time > JOINT_STATE_TIMEOUT:
            self.get_logger().error(f'No joint state messages received for {JOINT_STATE_TIMEOUT} seconds. Exiting...')
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
            
def main(args=None):
    try:
        rclpy.init(args=args)
        gato_controller = GATO_Controller()
        rclpy.spin(gato_controller)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_controller.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()