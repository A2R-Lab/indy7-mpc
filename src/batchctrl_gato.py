import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import sys 
import os
sys.path.append('../../') # to get bindings
import bindings.batch_sqp as batch_sqp

# set seed
np.random.seed(123)

# Add timeout duration in seconds
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

class TorqueCalculator(Node):
    def __init__(self):
        super().__init__('torque_calculator')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            1)
        # self.goal_sub = self.create_subscription(
        #     PoseStamped,
        #     'goal',
        #     self.goal_callback,
        #     10
        # )
        self.publisher = self.create_publisher(
            JointState, 
            'joint_commands', 
            10)
        self.ctrl_msg = JointState()
        self.ctrl_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.jointstate_count = 0
        

        # robot properties
        self.nq = 6
        self.nv = 6
        self.nu = 6
        self.nx = self.nq + self.nv
        

        # sim parameters
        self.batch_size = 1
        self.fext_timesteps = 0
        self.resample_fext = False # if true, resample fexts around the best result
        self.constant_frc = True # if true, true external forces are constant, else they are sampled from a normal distribution around last fext
        # if self.constant_frc:
        #     self.getfrc = lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # else:
        #     self.lastfrc = np.random.normal(self.lastfrc, 2.0)
        #     self.getfrc = lambda: list(self.lastfrc)
        self.lastfrc = np.zeros(self.nu) # true external forces

        # solver parameters
        self.N = 32
        self.dt = 0.01
        # dQ_cost = 1e-3
        # R_cost = 1e-5
        # QN_cost = 0.1
        # Qlim_cost = 1e-4
        self.solver = batch_sqp.SQPSolverfloat_1()

        self.config = f'{self.batch_size}_{self.fext_timesteps}_{self.resample_fext}_{self.constant_frc}'

        self.last_state_msg = None

        self.fig8 = figure8()
        self.fig8_offset = 0

        

        self.eepos_g = np.array(self.fig8[self.fig8_offset:self.fig8_offset+(6*self.N)], dtype=float)
        self.eepos_g_batch = np.tile(self.eepos_g, (self.batch_size, 1))


        self.f_ext_batch = np.zeros((self.batch_size, 6))
        self.solver.set_external_wrench_batch(self.f_ext_batch)

        # self.fext_batch = 50.0 * np.ones((self.batch_size, 3))
        # self.fext_batch = np.random.normal(0.0, 10.0, (self.batch_size, 3))
        # self.fext_batch[0] = np.array([0.0, 0.0, 0.0])
        # print(self.fext_batch)

        self.last_xs = None
        self.last_u = None

    
        # stats
        self.tracking_errs = []
        self.positions = []

        # Create a timer to check for timeout
        self.timeout_timer = self.create_timer(1.0, self.check_timeout)
        self.last_joint_state_time = time.time()

        self.XU = np.zeros(self.N*(self.nx+self.nu)-self.nu, dtype=float)
        self.XU_batch = np.tile(self.XU, (self.batch_size, 1))
        self.xs_batch = np.tile(np.zeros(self.nx), (self.batch_size, 1))
    
    # def goal_callback(self, msg):
    #     print('received pose: ', np.array(msg.pose.position))
    #     self.goal_trace = np.tile(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 0.0, 0.0, 0.0]), self.t.N)
    #     self.eepos_g_batch = [self.goal_trace] * self.batch_size

    def joint_callback(self, msg):
        
        print("called")
        print(self.fig8[self.fig8_offset:self.fig8_offset+6*self.N])
        self.last_joint_state_time = time.time()
        self.jointstate_count += 1

        xs = np.hstack([np.array(msg.position, dtype=float), np.array(msg.velocity, dtype=float)])
        print(xs)
        self.solver.reset()
        self.solver.resetRho()
        #self.solver.resetLambda()

        # update xs
        self.XU_batch[:,0:self.nx] = xs
        self.xs_batch[:] = xs
        s = time.time()
        # print all shapes
        print(f'xs_batch shape: {self.xs_batch.shape}')
        print(f'eepos_g_batch shape: {self.eepos_g_batch.shape}')
        print(f'XU_batch shape: {self.XU_batch.shape}')
        result = self.solver.solve(self.XU_batch, self.dt, self.xs_batch, self.eepos_g_batch)
        XU_batch_res = result["xu_trajectory"]
        print(f"batch sqp time: {1000 * (time.time() - s)} ms")
        step_sizes = []
        pcg_iters = []
        for i in range(len(result["line_search_stats"])):
            step_sizes.append(result["line_search_stats"][i]["step_size"])
        print(f'step sizes: {step_sizes}')
        for i in range(len(result["pcg_stats"])):
            pcg_iters.append(result["pcg_stats"][i]["pcg_iterations"])
        print(f'pcg iters: {pcg_iters}')
        
        
        # if 0 and self.last_state_msg is not None:
        #     # TODO: implement this

        #     m1_time = self.last_state_msg.header.stamp.sec + self.last_state_msg.header.stamp.nanosec * 1e-9
        #     m2_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        #     step_duration = m2_time - m1_time
        #     # get prediction based on last applied control # TODO: implement this
        #     predictions = self.t.predict_fwd(self.last_xs, self.last_u, step_duration)

        #     best_tracker_idx = None
        #     best_error = np.inf
        #     for i, result in enumerate(predictions):
        #         # get expected state for each result
        #         error = np.linalg.norm(result - self.xs)
        #         if error < best_error:
        #             best_error = error
        #             best_tracker_idx = i
            
        #     # resample fexts around the best result
        #     if self.resample_fext:
        #         self.fext_batch[:] = self.fext_batch[best_tracker_idx]
        #         self.fext_batch = np.random.normal(self.fext_batch, 2.0)
        #         self.t.batch_set_fext(self.fext_batch)
        # else:
        best_tracker_idx = 0

        #print(f'most accurate force: {self.f_ext_batch[best_tracker_idx]}')
        best_result = XU_batch_res[best_tracker_idx]
        effort = best_result[self.nx:(self.nx+self.nu)]
        print(f'effort: {effort}')

        # Publish torques from batch result that best matched dynamics on the last step
        self.ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.ctrl_msg.position = [0.0] * self.nq # list(best_result[:self.nq])
        self.ctrl_msg.velocity = [0.0] * 6
        self.ctrl_msg.effort = [float(e) for e in effort]
        self.publisher.publish(self.ctrl_msg)


        # # record stats
        # eepos = self.t.eepos(self.xs[0:self.t.nq])
        # self.positions.append(eepos)
        # self.tracking_errs.append(np.linalg.norm(eepos - self.goal_trace[:3]))
        # if self.jointstate_count % 400 == 0:
        #     # save tracking err to a file
        #     np.save(f'data/tracking_errs_{self.config}.npy', np.array(self.tracking_errs))
        #     np.save(f'data/positions_{self.config}.npy', np.array(self.positions))
        
        # fill in variables for next iteration
        self.last_xs = xs
        self.last_u = best_result[self.nx:(self.nx+self.nu)]
        self.last_state_msg = msg
        self.XU_batch[:] = XU_batch_res[best_tracker_idx]

        # shift the goal trace
        self.fig8_offset += 6
        self.fig8_offset %= len(self.fig8)
        copy1_len = min(self.fig8_offset + 6*self.N, len(self.fig8)) - self.fig8_offset
        copy2_len = 6*self.N - copy1_len
        self.eepos_g_batch[:,:copy1_len] = self.fig8[self.fig8_offset:self.fig8_offset+copy1_len]
        if copy2_len > 0:
            self.eepos_g_batch[:,-copy2_len:] = self.fig8[:copy2_len]

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
        torque_calculator = TorqueCalculator()
        rclpy.spin(torque_calculator)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        torque_calculator.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
