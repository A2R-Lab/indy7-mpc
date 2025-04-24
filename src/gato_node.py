import numpy as np
import sys
import os
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray
from batch_solver import GATO_Batch_Solver


# timeout duration for joint state subscriber
JOINT_STATE_TIMEOUT = 10.0
# interval for saving statistics (in seconds)
STATS_SAVE_INTERVAL = 35.0

class GATO_Node(Node):
    def __init__(self, batch_size=1, N=32, dt=0.01, 
                 f_ext_std=0.0, f_ext_resample_std=0.0):
        
        super().__init__('GATO_Node')


        self.batch_size = batch_size
        self.N = N
        self.dt = dt
        self.nu, self.nx = 6, 12
        self.config = f"dt:{self.dt}_batch_size:{self.batch_size}_N:{self.N}_f_ext_std:{f_ext_std}"

        self.reference_traj_sub = self.create_subscription(Float64MultiArray, '/ee_ref_traj', self.reference_traj_callback, 1)
        self.subscription = self.create_subscription(JointState, 'joint_states', self.joint_callback, 1)
        
        self.publisher = self.create_publisher(JointState, 'joint_commands', 1)
        
        self.external_force_publisher = self.create_publisher(Wrench, 'external_force', 1)
        
        self.latest_traj = np.tile(np.array([1.0, 0.5, 0.5, 0.0, 0.0, 0.0]), N)
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.ctrl_msg.position = [0.0] * 6 
        self.ctrl_msg.velocity = [0.0] * 6
        
        self.solver = GATO_Batch_Solver(self.N, self.dt, self.batch_size, f_ext_std=f_ext_std, f_ext_resample_std=f_ext_resample_std)
    
        # initialize inputs
        self.ee_pos_traj_batch = np.zeros((self.batch_size, self.N*6))        
        XU_batch = np.tile(np.zeros(N*18-6), (self.solver.batch_size, 1))
        self.xs_batch = np.tile(np.zeros(12), (self.solver.batch_size, 1))
        self.XU_batch, _ = self.solver.solve(self.xs_batch, self.ee_pos_traj_batch, XU_batch)
        self.XU_best = self.XU_batch[0, :]
        self.x_last, self.u_last = None, None
        
        # stats
        self.joint_positions = []
        self.solve_times = []
    
        # timing
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        self.last_time = time.time()
        self.stats_save_timer = self.create_timer(STATS_SAVE_INTERVAL, self.save_stats)
        self.last_save_time = time.time()
        self.last_log_time = time.time()
        
    def reference_traj_callback(self, msg):
        self.latest_traj = np.array(msg.data)
        
    def joint_callback(self, msg):
        x_curr = np.hstack([np.array(msg.position), np.array(msg.velocity)])
        ee_pos = np.array(msg.effort[0:3])
        
        if self.x_last is None:
            self.x_last, self.u_last = x_curr, np.zeros(6)
        
        # timing
        elapsed_time = time.time() - self.last_time
        self.last_time = time.time()
        
        # ----- update batched inputs -----
        self.ee_pos_traj_batch[:,:] = self.latest_traj
        self.xs_batch[:,:] = x_curr
        self.XU_batch[:, 0:self.nx] = x_curr
        
        # ----- solve -----
        self.solver.reset()
        self.solver.reset_rho()
        self.solver.solver.resetLambda()
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
        self.joint_positions.append(x_curr[:6])
        self.solve_times.append(solve_time)
        
        self.XU_batch[:,:] = self.XU_best
        self.x_last, self.u_last = x_curr, u_curr
        
        if time.time() - self.last_log_time > 0.1:
            self.last_log_time = time.time()
            self.get_logger().info(f"best idx: {best_idx}")
            self.get_logger().info(f"error: {np.linalg.norm(self.ee_pos_traj_batch[:, :3] - self.xs_batch[:, :3])}")
            self.get_logger().info(f"f_ext: {[f'{x:.4f}' for x in self.solver.f_ext_batch[best_idx][:3]]}")
            #self.get_logger().info(f"q: {[f'{x:3.2f}' for x in x_curr[:6]]}")
            self.get_logger().info(f"u: {[f'{x:.2f}' for x in u_curr]}")
            #self.get_logger().info(f"time elapsed: {elapsed_time:.3f}s")
            self.get_logger().info("\n--------------------------------\n")
        
    def save_stats(self):
        current_time = time.time()
        elapsed_since_last_save = current_time - self.last_save_time
        
        if elapsed_since_last_save >= STATS_SAVE_INTERVAL:
            self.last_save_time = current_time

            timestamp = datetime.now().strftime("%H%M%S")
            base_filename = os.path.join("stats/", f"{timestamp}")
            
            # Convert lists to numpy arrays
            joint_positions = np.array(self.joint_positions)
            solve_times = np.array(self.solve_times)
            
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

        dt = 0.01
        N = 32
        ref_traj = figure_8(A_x=0.5, A_z=0.55, 
                    offset=[0.0, 0.4, 0.45], 
                    period=10, 
                    dt=dt, 
                    cycles=10)
        
        padding = np.tile(ref_traj[0:6], 200)
        ref_traj = np.concatenate([padding, ref_traj])
        
        # # ----- Single solver config -----
        
        # gato_controller = GATO_Controller(ref_traj=ref_traj, batch_size=1, N=N, dt=dt,
        #                                   f_ext_std=0.0, f_ext_resample_std=0.0, 
        #                                   f_ext_actual=f_ext_actual)
        
        # ----- 32 BATCH SAMPLE CONFIG -----

        # gato_controller = GATO_Controller(ref_traj=ref_traj, batch_size=32, N=N, dt=dt,
        #                                   f_ext_std=2.0, f_ext_resample_std=0.2, 
        #                                   f_ext_actual=f_ext_actual)
        
        # # ----- 64 BATCH SAMPLE CONFIG -----

        gato_controller = GATO_Controller(batch_size=32, N=N, dt=dt,
                                          f_ext_std=20.0, f_ext_resample_std=1.0)
        
        rclpy.spin(gato_controller)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_controller.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()