import numpy as np
from datetime import datetime
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
# interval for saving statistics (in seconds)
STATS_SAVE_INTERVAL = 15.0

def figure8(A_x=0.4, A_z=0.4, offset=[0.0, 0.5, 0.6], period=5, dt=0.01, cycles=1):
    x = lambda t: offset[0] + A_x * np.sin(t)  # X goes from -xamplitude to xamplitude
    y = lambda t: offset[1]
    z = lambda t: offset[2] + A_z * np.sin(2*t)/2 + A_z/2  # Z goes from -zamplitude/2 to zamplitude/2
    
    timesteps = np.linspace(0, 2*np.pi, int(period/dt))
    fig_8 = np.array([[x(t), y(t), z(t), 0.0, 0.0, 0.0] for t in timesteps]).reshape(-1)
    return np.tile(fig_8, cycles)

class GATO_Batch_Sample:
    def __init__(self, N=32, dt=0.01, batch_size=4, stats=None, f_ext_std=1.0, f_ext_resample_std=0.0):
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
            self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        else:
            self.f_ext_batch = np.zeros((self.batch_size, 6))

        self.resample_f_ext = True if f_ext_resample_std > 0.0 else False
        self.f_ext_resample_std = f_ext_resample_std
        
        print("f_ext_batch:")
        print(np.array2string(self.f_ext_batch, precision=4, suppress_small=True))
        self.solver.set_external_wrench_batch(self.f_ext_batch)

        
    def solve(self, xcur_batch, eepos_goals_batch, XU_batch):
        
        result = self.solver.solve(XU_batch, self.dt, xcur_batch, eepos_goals_batch)
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
            self.f_ext_batch = np.random.normal(self.f_ext_batch, self.f_ext_resample_std, self.f_ext_batch.shape)
            self.f_ext_batch[:, 3:] = 0.0
            self.f_ext_batch[best_idx] = f_ext_best
            self.f_ext_batch[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.solver.set_external_wrench_batch(self.f_ext_batch)

    def reset(self):
        self.solver.reset()
        
    def reset_lambda(self):
        self.solver.resetLambda()
    
    def reset_rho(self):
        self.solver.resetRho()
        
    def get_stats(self):
        return self.stats


class GATO_Controller(Node):
    def __init__(self, ref_traj=None, batch_size=1, N=32, dt=0.01, 
                 f_ext_std=0.0, f_ext_resample_std=0.0, 
                 f_ext_actual=None):
        
        super().__init__('gato_controller')
        
        if ref_traj is None:
            raise ValueError("ref_traj must be provided")
        self.ref_traj = ref_traj
        self.ref_traj_offset = 0

        self.batch_size = batch_size
        self.N = N
        self.dt = dt
        self.nu, self.nx = 6, 12
        self.config = f"dt:{self.dt}_batch_size:{self.batch_size}_N:{self.N}_f_ext_std:{f_ext_std}"


        self.subscription = self.create_subscription(JointState, 'joint_states', self.joint_callback, 1)
        
        self.publisher = self.create_publisher(JointState, 'joint_commands', 1)
        
        self.external_force_publisher = self.create_publisher(Wrench, 'external_force', 1)
        
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.ctrl_msg.position = [0.0] * 6 
        self.ctrl_msg.velocity = [0.0] * 6
        
        self.solver = GATO_Batch_Sample(self.N, self.dt, self.batch_size, f_ext_std=f_ext_std, f_ext_resample_std=f_ext_resample_std)

        f_ext_actual = f_ext_actual if f_ext_actual is not None else np.zeros(6)
        self.send_external_force(f_ext_actual)
    
        # initialize inputs
        self.ee_pos_traj_batch = np.tile(self.ref_traj[:6*self.N], (self.batch_size, 1))
        XU_batch = np.tile(np.zeros(N*18-6), (self.solver.batch_size, 1))
        self.xs_batch = np.tile(np.zeros(12), (self.solver.batch_size, 1))
        self.XU_batch, _ = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, XU_batch)
        self.XU_best = self.XU_batch[0, :]
        self.x_last, self.u_last = None, None
        
        # stats
        self.dts = []
        self.tracking_errors = []
        self.ee_positions = []
        self.ee_ref_positions = []
        self.joint_positions = []
        self.solve_times = []
    
        # timing
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        self.last_time = time.time()
        self.stats_save_timer = self.create_timer(STATS_SAVE_INTERVAL, self.save_stats)
        self.last_save_time = time.time()
        
    def joint_callback(self, msg):
        x_curr = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        ee_pos = np.array(msg.effort[0:3])
        
        if self.x_last is None:
            self.x_last, self.u_last = x_curr, np.zeros(6)
        
        # timing
        elapsed_time = time.time() - self.last_time
        self.dts.append(elapsed_time)
        self.last_time = time.time()
        
        # ----- update batched inputs -----
        self.ref_traj_offset += elapsed_time/self.dt
        offset = int(self.ref_traj_offset)
        self.ee_pos_traj_batch[:,:] = self.ref_traj[6*offset:6*(offset+self.N)]
        self.xs_batch[:,:] = x_curr
        self.XU_batch[:, 0:self.nx] = x_curr
        
        # ----- solve -----
        self.solver.reset()
        #self.solver.solver.resetRho()
        #self.solver.solver.resetLambda()
        XU_batch_new, solve_time = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, self.XU_batch)
        best_idx = self.solver.find_best_idx(self.x_last, self.u_last, x_curr, elapsed_time)
        self.solver.resample_f_ext_batch(best_idx)
        self.XU_best = XU_batch_new[best_idx]
        u_curr = self.XU_best[self.nx:(self.nx+self.nu)]
        
        # ----- publish control -----
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.effort = [float(e) for e in u_curr]
        self.publisher.publish(self.ctrl_msg)
        
        # ----- log stats and update -----
        tracking_error = np.linalg.norm(ee_pos - self.ee_pos_traj_batch[0, :3])
        self.tracking_errors.append(tracking_error)
        self.ee_positions.append(ee_pos)
        self.ee_ref_positions.append(self.ee_pos_traj_batch[0, :3])
        self.joint_positions.append(x_curr[:6])
        self.solve_times.append(solve_time)
        
        self.XU_batch[:,:] = self.XU_best
        self.x_last, self.u_last = x_curr, u_curr
        
        print(f"best idx: {best_idx}  ----- f_ext: {[f'{x:.4f}' for x in self.solver.f_ext_batch[best_idx][:3]]}")
        print(f"q: {[f'{x:.4f}' for x in x_curr[:6]]}")
        print(f"u: {[f'{x:.4f}' for x in u_curr]}")
        print(f"time elapsed: {elapsed_time:.4f}s")
        print("\n--------------------------------\n")
        
    def send_external_force(self, wrench):
        self.f_ext_actual = wrench
        wrench_msg = Wrench()
        wrench_msg.force.x = float(wrench[0])
        wrench_msg.force.y = float(wrench[1])
        wrench_msg.force.z = float(wrench[2])
        wrench_msg.torque.x = float(wrench[3])
        wrench_msg.torque.y = float(wrench[4])
        wrench_msg.torque.z = float(wrench[5])
        self.external_force_publisher.publish(wrench_msg)
        self.get_logger().info(f'Sent external wrench: {wrench}')

    def save_stats(self):
        current_time = time.time()
        elapsed_since_last_save = current_time - self.last_save_time
        
        if elapsed_since_last_save >= STATS_SAVE_INTERVAL:
            self.last_save_time = current_time

            timestamp = datetime.now().strftime("%H%M%S")
            base_filename = os.path.join("stats/", f"{timestamp}")
            
            # Convert lists to numpy arrays
            dts = np.array(self.dts)
            tracking_errors = np.array(self.tracking_errors)
            ee_positions = np.array(self.ee_positions)
            ee_ref_positions = np.array(self.ee_ref_positions)
            joint_positions = np.array(self.joint_positions)
            solve_times = np.array(self.solve_times)
            
            # Save each array to a separate .npy file
            np.save(f"{base_filename}_dts.npy", dts)
            np.save(f"{base_filename}_tracking_errors.npy", tracking_errors)
            np.save(f"{base_filename}_ee_positions.npy", ee_positions)
            np.save(f"{base_filename}_ee_ref_positions.npy", ee_ref_positions)
            np.save(f"{base_filename}_joint_positions.npy", joint_positions)
            np.save(f"{base_filename}_solve_times.npy", solve_times)
            
            self.get_logger().info(f'Statistics saved to {base_filename}_*.npy files')
            
        
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

        dt = 0.03
        N = 32
        f_ext_actual = [10.0, 20.0, -30.0, 0.0, 0.0, 0.0]
        ref_traj = figure8(A_x=0.4, A_z=0.4, 
                           offset=[0.0, 0.5, 0.6], 
                           period=10, 
                           dt=dt, 
                           cycles=10)
        
        # ----- 32 BATCH SAMPLE CONFIG -----

        gato_controller = GATO_Controller(ref_traj=ref_traj, batch_size=64, N=N, dt=dt,
                                          f_ext_std=2.0, f_ext_resample_std=0.5, 
                                          f_ext_actual=f_ext_actual)
        
        # ----- 64 BATCH SAMPLE CONFIG -----

        # gato_controller = GATO_Controller(ref_traj=ref_traj, batch_size=64, N=N, dt=dt,
        #                                   f_ext_std=5.0, f_ext_resample_std=1.0, 
        #                                   f_ext_actual=f_ext_actual)
        
 
        rclpy.spin(gato_controller)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_controller.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()