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
from geometry_msgs.msg import PoseStamped, Wrench

np.random.seed(42)
# timeout duration for joint state subscriber
JOINT_STATE_TIMEOUT = 10.0

def figure8(A_x=0.4, A_z=0.4, offset=[0.0, 0.50, 0.7], period=5, dt=0.01, cycles=1):
    x = lambda t: offset[0] + A_x * np.sin(t)  # X goes from -xamplitude to xamplitude
    y = lambda t: offset[1]
    z = lambda t: offset[2] + A_z * np.sin(2*t)/2 + A_z/2  # Z goes from -zamplitude/2 to zamplitude/2
    
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    fig_8 = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    return np.tile(fig_8, cycles)

class GATO_Batch_Sample:
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
        # if f_ext_std != 0.0:
        self.f_ext_batch = np.random.normal(0, f_ext_std, (self.batch_size, 6))
        self.f_ext_batch[:, 3:] = 0.0
        self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        # else:
        #     self.f_ext_batch = np.zeros((self.batch_size, 6))
        
        print("f_ext_batch:")
        print(self.f_ext_batch)
        self.solver.set_external_wrench_batch(self.f_ext_batch)
        
        
        
    def solve(self, xcur_batch, eepos_goals_batch, XU_batch):
        # xcur_batch: (batch_size, state_size)
        # eepos_goals_batch: (batch_size, N * 6)
        # XU_batch: (batch_size, N*(state_size+control_size)-control_size)
        
        result = self.solver.solve(XU_batch, self.dt, xcur_batch, eepos_goals_batch)

        self.stats['solve_time']['values'].append(result["solve_time_us"])
        self.stats['sqp_iters']['values'].append(result["sqp_iterations"])
        for i in range(len(result["pcg_stats"])):
            self.stats['pcg_iters']['values'].append(result["pcg_stats"][i]["pcg_iterations"])
        for i in range(len(result["line_search_stats"])):
            self.stats["step_size"]["values"].append(result["line_search_stats"][i]["step_size"])
            
        return result["xu_trajectory"]
    
    def sim_forward(self, xk, uk, sim_dt):
        x_next_batch = self.solver.sim_forward(xk, uk, sim_dt)
        return x_next_batch
    
    def reset(self):
        self.solver.reset()
        
    def reset_lambda(self):
        self.solver.resetLambda()
    
    def reset_rho(self):
        self.solver.resetRho()
        
    def get_stats(self):
        return self.stats





class GATO_Controller(Node):
    def __init__(self, ref_traj=None, batch_size=1, N=32, dt=0.01, f_ext_std=0.0, f_ext_resample_std=0.5, f_ext_actual=None):
        
        if ref_traj is None:
            raise ValueError("ref_traj must be provided")
        self.ref_traj = ref_traj
        self.ref_traj_offset = 0
        
        super().__init__('gato_controller')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            1)
        
        self.publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            1)
        
        self.external_force_publisher = self.create_publisher(
            Wrench,
            'external_force',
            1)
        
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        self.batch_size = batch_size
        self.N = N
        self.dt = dt
        self.nu, self.nx = 6, 12
        self.config = f"dt:{self.dt}_batch_size:{self.batch_size}_N:{self.N}_f_ext_std:{f_ext_std}"
        
        self.solver = GATO_Batch_Sample(self.N, self.dt, self.batch_size, f_ext_std=f_ext_std)
        self.f_ext_resample_std = f_ext_resample_std
    
        # initialize
        self.ee_pos_traj_batch = np.tile(self.ref_traj[:6*self.N], (self.batch_size, 1))
        XU_batch = np.tile(np.zeros(N*18-6), (self.solver.batch_size, 1))
        self.xs_batch = np.tile(np.zeros(12), (self.solver.batch_size, 1))
        self.XU_batch = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, XU_batch)
        self.XU_best = self.XU_batch[0, :]
        
        self.x_last = None
        self.u_last = None
        self.resample_f_ext = True
        
        # stats
        self.delta_ts = []
        self.tracking_errors = []
        self.ee_positions = []
        self.joint_positions = []
        
        if f_ext_actual is not None:
            self.f_ext = f_ext_actual
        else:
            self.f_ext = np.zeros(6)
            
        self.send_external_force()
        
        # timing
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        self.last_time = time.time()
        
        
    def find_best_trajectory_id(self, x_last, u_last, xs, dt):
        best_error = np.inf
        best_idx = None
        
        # self.solver.solver.set_external_wrench_batch(self.solver.f_ext_batch)
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        for i in range(self.solver.batch_size):
            x_next = x_next_batch[i, :]
            error = np.linalg.norm(x_next - xs)
            
            if error <= best_error:
                best_error = error
                best_idx = i
                
        if self.resample_f_ext:
            f_ext_best = self.solver.f_ext_batch[best_idx, :]
            self.solver.f_ext_batch = np.tile(f_ext_best, (self.solver.batch_size, 1))
            self.solver.f_ext_batch += np.random.normal(0, self.f_ext_resample_std, self.solver.f_ext_batch.shape)
            self.solver.f_ext_batch[:, 3:] = 0.0
            self.solver.f_ext_batch[best_idx] = f_ext_best
            self.solver.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            self.solver.solver.set_external_wrench_batch(self.solver.f_ext_batch)
                
        return best_idx
        
    def joint_callback(self, msg):
        if self.x_last is None:
            self.x_last = np.hstack([np.array(msg.position), np.array(msg.velocity)])
            self.u_last = np.zeros(6)
        
        # timing
        elapsed_time = time.time() - self.last_time
        self.delta_ts.append(elapsed_time)
        self.last_time = time.time()
        self.ref_traj_offset += elapsed_time/self.dt
        offset = int(self.ref_traj_offset)
        
        # update batched states
        xs = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        self.xs_batch[:,:] = xs
        
        # update batched ee pos goal
        self.ee_pos_traj_batch[:,:] = self.ref_traj[6*offset:6*(offset+self.N)]
        
        # update batched XU
        self.XU_batch[:,:] = self.XU_best
        self.XU_batch[:, 0:self.nx] = xs
        print(f"xs: {xs}")
        
        # solve
        start_time = time.time()
        self.solver.reset()
        #self.solver.solver.resetRho()
        #self.solver.solver.resetLambda()
        XU_batch_new = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, self.XU_batch)
        end_time = time.time()
        #print(f"batch sqp time: {1000 * (end_time - start_time)} ms")
        #print(f"step sizes: {step_sizes}")
        
        best_idx = self.find_best_trajectory_id(self.x_last, self.u_last, xs, elapsed_time)
        self.XU_best = XU_batch_new[best_idx]
        u = self.XU_best[self.nx:(self.nx+self.nu)]
        print(f"u: {u}")
        print(f"best_idx: {best_idx}")
        
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * 6 
        self.ctrl_msg.velocity = [0.0] * 6
        self.ctrl_msg.effort = [float(e) for e in u]
        self.publisher.publish(self.ctrl_msg)
        
        
        # self.fig8_offset += 6
        # self.fig8_offset %= len(self.fig8)
        # copy1_len = min(self.fig8_offset + 6*self.N, len(self.fig8)) - self.fig8_offset
        # copy2_len = 6*self.N - copy1_len
        # self.eepos_g_batch[:,:copy1_len] = self.fig8[self.fig8_offset:self.fig8_offset+copy1_len]
        # if copy2_len > 0:
        #     self.eepos_g_batch[:,-copy2_len:] = self.fig8[:copy2_len]
            
        self.x_last = xs
        self.u_last = u
        
        print("\n--------------------------------\n\n")
        
        
    def send_external_force(self):
        force = self.f_ext[0:3]
        torque = self.f_ext[3:6]
        
        wrench_msg = Wrench()
        wrench_msg.force.x = float(force[0])
        wrench_msg.force.y = float(force[1])
        wrench_msg.force.z = float(force[2])
        wrench_msg.torque.x = float(torque[0])
        wrench_msg.torque.y = float(torque[1])
        wrench_msg.torque.z = float(torque[2])
        
        self.external_force_publisher.publish(wrench_msg)
        self.get_logger().info(f'Sent external force: {force}, torque: {torque}')
        
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
        
        ref_traj = figure8(A_x=0.4, A_z=0.4, offset=[0.0, 0.50, 0.7], period=5, dt=0.01, cycles=5)
        
        f_ext_actual = [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
        
        gato_controller = GATO_Controller(ref_traj=ref_traj, batch_size=32, N=32, dt=0.01,
                                          f_ext_std=0.0, f_ext_actual=f_ext_actual)
        rclpy.spin(gato_controller)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_controller.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()