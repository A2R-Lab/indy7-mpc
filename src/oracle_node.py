import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from reference_traj import figure_8
from geometry_msgs.msg import Wrench, PointStamped, Point, Vector3
from sensor_msgs.msg import JointState
import time
import os



class OracleNode(Node):
    def __init__(self, N=32, dt=0.02, f_ext_update_std=0.0, f_ext_max=100.0):
        super().__init__('traj_oracle_node')
        self.N = N
        self.dt = dt
        self.traj_pub = self.create_publisher(Float64MultiArray, '/ee_ref_traj', 1)
        self.timer = self.create_timer(self.dt, self.publish_trajectory_callback)
        self.traj_offset = 0
        self.external_force_pub = self.create_publisher(Wrench, '/external_force', 1)
        self.f_ext_update_period = 10.0
        self.f_ext_timer = self.create_timer(self.f_ext_update_period, self.publish_external_force_callback)
        self.f_ext_actual = np.array([-30.0, 10.0, -30.0])
        self.f_ext_max = f_ext_max
        self.f_ext_update_std = f_ext_update_std
        
        # Generate reference trajectory 
        self.ref_traj = figure_8(
            x_amplitude=0.4, z_amplitude=0.5,
            offset=[0.0, 0.6, 0.5],
            timestep=dt, period=10, num_periods=10, angle_offset=np.pi/4
        )
        padding = np.tile(self.ref_traj[0:6], 200)
        self.ref_traj = np.concatenate([padding, self.ref_traj])
        self.max_offset = len(self.ref_traj) // 6 - N

        # Error tracking
        self.tracking_errors = []
        self.ee_positions = []
        self.ee_ref_positions = []
        
        self.stats_save_period = 30.0  # seconds
        self.stats_timer = self.create_timer(self.stats_save_period, self.save_stats) # save_stats() callback

        # Subscribe to end-effector position
        self.ee_pos_sub = self.create_subscription(
            PointStamped, 'ee_position', self.ee_pos_callback, 1
        )

        # Publishers for real-time plotting
        self.rt_ee_pos_pub = self.create_publisher(PointStamped, '/ee_pos_current', 10)
        self.rt_ee_ref_pub = self.create_publisher(Point, '/ee_ref_current', 10)
        self.rt_error_pub = self.create_publisher(Float64, '/tracking_error_current', 10)
        self.rt_force_pub = self.create_publisher(Vector3, '/external_force_vector', 10)

    def publish_trajectory_callback(self):
        # Compute current offset
        offset = int(self.traj_offset)
        if offset > self.max_offset:
            offset = 0  # Loop
            self.traj_offset = 0
        # Extract (6, N) segment
        traj_segment = self.ref_traj[6*offset:6*(offset+self.N)].reshape(6, self.N)
        msg = Float64MultiArray()
        msg.data = traj_segment.flatten().tolist()
        self.traj_pub.publish(msg)
        self.traj_offset += 1  # Advance by one step each call
        self.current_traj_segment = traj_segment  # Store for error tracking

    def publish_external_force_callback(self):
        noise = np.random.normal(0, self.f_ext_update_std, size=3)
        self.f_ext_actual[:3] = np.clip(self.f_ext_actual[:3] + noise, -self.f_ext_max, self.f_ext_max)
        wrench_msg = Wrench()
        wrench_msg.force.x = float(self.f_ext_actual[0])
        wrench_msg.force.y = float(self.f_ext_actual[1])
        wrench_msg.force.z = float(self.f_ext_actual[2])
        # just forces for now
        wrench_msg.torque.x = 0.0
        wrench_msg.torque.y = 0.0
        wrench_msg.torque.z = 0.0
        self.external_force_pub.publish(wrench_msg)
        self.get_logger().info(f"Published external wrench: {self.f_ext_actual}")

    def ee_pos_callback(self, msg: PointStamped):
        # Determine the current time step index. Since traj_offset is incremented
        # after publishing the segment for the *next* N steps, the relevant
        # reference point for the *current* ee_pos corresponds to the previous step index.
        current_step_index = int(self.traj_offset) - 1

        # Check if the index is valid and if we have started publishing trajectories
        if current_step_index < 0 or (current_step_index * 6 + 3) > len(self.ref_traj):
            # Log a warning or just return if it's too early or index is out of bounds
            self.get_logger().warn(f"Skipping error calculation: Invalid step index {current_step_index}.")
            return

        # Extract current EE position from PointStamped message
        ee_pos = np.array([msg.point.x, msg.point.y, msg.point.z])

        # Extract the corresponding reference position from the original full trajectory
        start_idx = 6 * current_step_index
        ee_ref = self.ref_traj[start_idx : start_idx + 3]

        # Calculate and store error
        tracking_error = np.linalg.norm(ee_pos - ee_ref)
        self.tracking_errors.append(tracking_error)
        self.ee_positions.append(ee_pos)
        self.ee_ref_positions.append(ee_ref)
        # Reduced logging frequency might be desirable here
        self.get_logger().info(f"[Current idx: {current_step_index}]")
        self.get_logger().info(f"F_ext: {self.f_ext_actual}")
        self.get_logger().info(f"Ref: {ee_ref[0]:.3f}, {ee_ref[1]:.3f}, {ee_ref[2]:.3f}")
        self.get_logger().info(f"Actual: {ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}")
        self.get_logger().info(f"Error: {tracking_error:.4f}")
        self.get_logger().info(f"--------------------------------\n")
        
        # Publish data for real-time plotting
        # Republish incoming ee_pos with potentially new timestamp if needed, or just pass it through
        self.rt_ee_pos_pub.publish(msg) 
        
        ref_point_msg = Point()
        ref_point_msg.x = float(ee_ref[0])
        ref_point_msg.y = float(ee_ref[1])
        ref_point_msg.z = float(ee_ref[2])
        self.rt_ee_ref_pub.publish(ref_point_msg)
        
        error_msg = Float64()
        error_msg.data = float(tracking_error)
        self.rt_error_pub.publish(error_msg)
        
        # Publish current external force vector
        force_vec_msg = Vector3()
        force_vec_msg.x = float(self.f_ext_actual[0])
        force_vec_msg.y = float(self.f_ext_actual[1])
        force_vec_msg.z = float(self.f_ext_actual[2])
        self.rt_force_pub.publish(force_vec_msg)
        
    def save_stats(self):
        current_time = time.time()
        if len(self.tracking_errors) == 0:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join("stats/", f"oracle_{timestamp}")
        np.save(f"{base_filename}_tracking_errors.npy", np.array(self.tracking_errors))
        np.save(f"{base_filename}_ee_positions.npy", np.array(self.ee_positions))
        np.save(f"{base_filename}_ee_ref_positions.npy", np.array(self.ee_ref_positions))
        self.get_logger().info(f"[Oracle] Stats saved to {base_filename}_*.npy files")

def main(args=None):
    rclpy.init(args=args)
    node = OracleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
