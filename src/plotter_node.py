#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped, Point, Vector3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
import matplotlib.gridspec as gridspec # Import gridspec

class PlotterNode(Node):
    def __init__(self, max_points=100): 
        super().__init__('plotter_node')
        self.max_points = max_points

        # Data storage using deques for efficient appending and popping
        self.time_steps = deque(maxlen=self.max_points)
        self.errors = deque(maxlen=self.max_points)
        self.ee_pos_x = deque(maxlen=self.max_points)
        self.ee_pos_y = deque(maxlen=self.max_points)
        self.ee_pos_z = deque(maxlen=self.max_points)
        self.ee_ref_x = deque(maxlen=self.max_points)
        self.ee_ref_y = deque(maxlen=self.max_points)
        self.ee_ref_z = deque(maxlen=self.max_points)
        self.current_step = 0
        self.latest_force = None # Store latest force vector
        self.force_quiver_3d = None # Rename 3D quiver handle
        self.error_quiver_3d = None # Rename 3D quiver handle

        # Quiver handles for 2D views
        self.force_quiver_top = None
        self.error_quiver_top = None
        self.force_quiver_front = None
        self.error_quiver_front = None
        self.force_quiver_right = None
        self.error_quiver_right = None

        # Subscribers
        self.error_sub = self.create_subscription(
            Float64, '/tracking_error_current', self.error_callback, 10)
        self.ee_pos_sub = self.create_subscription(
            PointStamped, '/ee_pos_current', self.ee_pos_callback, 10)
        self.ee_ref_sub = self.create_subscription(
            Point, '/ee_ref_current', self.ee_ref_callback, 10)
        self.force_sub = self.create_subscription(
            Vector3, '/external_force_vector', self.force_callback, 10)

        # --- Setup Plots ---
        plt.ion() # Enable interactive mode
        self.fig = plt.figure(figsize=(10, 8)) # Adjusted figure size
        gs = gridspec.GridSpec(2, 3, figure=self.fig) # Use GridSpec for layout

        # Plot 1: Tracking Error (Top Left)
        self.ax_err = self.fig.add_subplot(gs[0, 0:2])
        self.err_line, = self.ax_err.plot([], [], 'r-', label='Tracking Error')
        self.ax_err.set_xlabel("Time Step")
        self.ax_err.set_ylabel("Error Norm")
        self.ax_err.set_title("Real-time Tracking Error")
        self.ax_err.legend()
        self.ax_err.grid(True)

        # Plot 2: 3D Trajectories (Top Middle/Right)
        self.ax_3d = self.fig.add_subplot(gs[0, 2], projection='3d')
        self.ee_line_3d, = self.ax_3d.plot([], [], [], 'b-', label='EE Position') # Rename line handle
        self.ref_line_3d, = self.ax_3d.plot([], [], [], 'r--', label='Reference') # Rename line handle
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_title("EE Trajectory")
        self.ax_3d.legend()
        self.ax_3d.view_init(elev=30., azim=135)
        # Initial limits (will auto-scale)
        self.ax_3d.set_xlim([-0.5, 0.5])
        self.ax_3d.set_ylim([0, 1.0])
        self.ax_3d.set_zlim([0, 1.0])

        # --- New 2D Views ---

        # Plot 3: Top View (X-Y) (Bottom Left)
        self.ax_top = self.fig.add_subplot(gs[1, 0], projection='3d', aspect='equal')
        self.ee_line_top, = self.ax_top.plot([], [], [], 'b-', label='EE (Top)')
        self.ref_line_top, = self.ax_top.plot([], [], [], 'r--', label='Ref (Top)')
        self.ax_top.set_xlabel("X")
        self.ax_top.set_ylabel("Y")
        self.ax_top.set_zlabel("Z")
        self.ax_top.set_title("")
        #self.ax_top.legend()
        self.ax_top.grid(True)
        self.ax_top.view_init(elev=90., azim=135)

        # Plot 4: Front View (Y-Z) (Bottom Middle)
        self.ax_front = self.fig.add_subplot(gs[1, 1], projection='3d', aspect='equal')
        self.ee_line_front, = self.ax_front.plot([], [], [], 'b-', label='EE (Front)')
        self.ref_line_front, = self.ax_front.plot([], [], [], 'r--', label='Ref (Front)')
        self.ax_front.set_xlabel("X")
        self.ax_front.set_ylabel("Y")
        self.ax_front.set_zlabel("Z")
        self.ax_front.set_title("")
        self.ax_front.set_xlim([-0.5, 0.5])
        self.ax_front.set_ylim([-0.5, 0.5])
        self.ax_front.set_zlim([-0.5, 0.5])
        #self.ax_front.legend()
        self.ax_front.grid(True)
        self.ax_front.view_init(elev=0., azim=135)
        # Plot 5: Right Side View (X-Z) (Bottom Right)
        self.ax_right = self.fig.add_subplot(gs[1, 2], projection='3d', aspect='equal')
        self.ee_line_right, = self.ax_right.plot([], [], [], 'b-', label='EE (Right)')
        self.ref_line_right, = self.ax_right.plot([], [], [], 'r--', label='Ref (Right)')
        self.ax_right.set_xlabel("X")
        self.ax_right.set_ylabel("Y")
        self.ax_right.set_zlabel("Z")
        self.ax_right.set_title("")
        #self.ax_right.legend()
        self.ax_right.grid(True)
        self.ax_right.view_init(elev=60., azim=45)
        
        plt.tight_layout() # Use tight_layout
        plt.show()

        # Timer for updating plots - might be smoother than updating in every callback
        self.plot_update_timer = self.create_timer(0.05, self.update_plots) # Update ~20 Hz

        self.get_logger().info(f"Plotter Node started. Listening for data...")

    def error_callback(self, msg):
        self.errors.append(msg.data)
        self.time_steps.append(self.current_step)
        # Increment step only when error is received, assuming it's the last value calculated
        self.current_step += 1

    def ee_pos_callback(self, msg): 
        self.ee_pos_x.append(msg.point.x)
        self.ee_pos_y.append(msg.point.y)
        self.ee_pos_z.append(msg.point.z)

    def ee_ref_callback(self, msg):
        self.ee_ref_x.append(msg.x)
        self.ee_ref_y.append(msg.y)
        self.ee_ref_z.append(msg.z)

    def force_callback(self, msg):
        self.latest_force = np.array([msg.x, msg.y, msg.z])

    def update_plots(self):
        if not self.time_steps: # No data yet
            return
            
        # Update error plot
        self.err_line.set_data(list(self.time_steps), list(self.errors))
        self.ax_err.relim()
        self.ax_err.autoscale_view(True, True, True)
        ymin, ymax = self.ax_err.get_ylim()
        self.ax_err.set_ylim(0, max(0.5, ymax))

        # Update 3D plot
        # Check if deque lengths match before plotting (they might not initially)
        min_len = min(len(self.ee_pos_x), len(self.ee_ref_x))
        if min_len > 0:
            ee_x = list(self.ee_pos_x)[-min_len:]
            ee_y = list(self.ee_pos_y)[-min_len:]
            ee_z = list(self.ee_pos_z)[-min_len:]
            ref_x = list(self.ee_ref_x)[-min_len:]
            ref_y = list(self.ee_ref_y)[-min_len:]
            ref_z = list(self.ee_ref_z)[-min_len:]
            
            # --- Update 3D Plot ---
            self.ee_line_3d.set_data(ee_x, ee_y)
            self.ee_line_3d.set_3d_properties(ee_z)
            self.ref_line_3d.set_data(ref_x, ref_y)
            self.ref_line_3d.set_3d_properties(ref_z)
            
            self.ee_line_front.set_data(ee_x, ee_y)
            self.ee_line_front.set_3d_properties(ee_z)
            self.ref_line_front.set_data(ref_x, ref_y)
            self.ref_line_front.set_3d_properties(ref_z)
            
            self.ee_line_top.set_data(ee_x, ee_y)
            self.ee_line_top.set_3d_properties(ee_z)
            self.ref_line_top.set_data(ref_x, ref_y)
            self.ref_line_top.set_3d_properties(ref_z)
            
            self.ee_line_right.set_data(ee_x, ee_y)
            self.ee_line_right.set_3d_properties(ee_z)
            self.ref_line_right.set_data(ref_x, ref_y)
            self.ref_line_right.set_3d_properties(ref_z)
            
            # Auto-scale 3D plot axes based on actual path
            if ee_x: # Check if list is not empty
                all_x = np.array(ee_x + ref_x)
                all_y = np.array(ee_y + ref_y)
                all_z = np.array(ee_z + ref_z)

                max_range = 1.2 * np.array([all_x.max()-all_x.min(), 
                                      all_y.max()-all_y.min(), 
                                      all_z.max()-all_z.min()]).max() / 2.0
                if max_range < 0.01: max_range = 0.1 # Prevent tiny scales at start
                    
                mid_x = (all_x.max()+all_x.min()) * 0.5
                mid_y = (all_y.max()+all_y.min()) * 0.5
                mid_z = (all_z.max()+all_z.min()) * 0.5
                self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
                
                self.ax_top.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax_top.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax_top.set_zlim(mid_z - max_range, mid_z + max_range)

                self.ax_front.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax_front.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax_front.set_zlim(mid_z - max_range, mid_z + max_range)
                
                self.ax_right.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax_right.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax_right.set_zlim(mid_z - max_range, mid_z + max_range)

                # --- Draw Error Vector ---
                # Remove previous error quiver (3D)
                if self.error_quiver_3d: self.error_quiver_3d.remove(); self.error_quiver_3d = None
                if self.error_quiver_top: self.error_quiver_top.remove(); self.error_quiver_top = None
                if self.error_quiver_front: self.error_quiver_front.remove(); self.error_quiver_front = None
                if self.error_quiver_right: self.error_quiver_right.remove(); self.error_quiver_right = None

                # Calculate latest error vector
                latest_ee_pos = np.array([ee_x[-1], ee_y[-1], ee_z[-1]])
                latest_ref_pos = np.array([ref_x[-1], ref_y[-1], ref_z[-1]])
                error_vec = latest_ref_pos - latest_ee_pos
                error_norm = np.linalg.norm(error_vec)
                error_scale_factor = 2.5 # Draw actual error magnitude

                if error_norm > 1e-4: # Only draw if error is non-negligible
                    pos = latest_ee_pos
                    # 3D Error Quiver
                    error_legend_exists_3d = any(artist.get_label() == 'Tracking Error Vector' for artist in self.ax_3d.collections + self.ax_3d.lines)
                    self.error_quiver_3d = self.ax_3d.quiver(
                        pos[0], pos[1], pos[2],
                        error_vec[0], error_vec[1], error_vec[2],
                        length=error_norm * error_scale_factor,
                        normalize=False,
                        color='m',
                        label='Tracking Error Vector' if not error_legend_exists_3d else ""
                    )
                    self.error_quiver_top = self.ax_top.quiver(
                        pos[0], pos[1], pos[2],
                        error_vec[0], error_vec[1], error_vec[2],
                        length=error_norm * error_scale_factor,
                        normalize=False,
                        color='m',
                        label='Tracking Error Vector' if not error_legend_exists_3d else ""
                    )                    
                    self.error_quiver_front = self.ax_front.quiver(
                        pos[0], pos[1], pos[2],
                        error_vec[0], error_vec[1], error_vec[2],
                        length=error_norm * error_scale_factor,
                        normalize=False,
                        color='m',
                    )   
                    self.error_quiver_right = self.ax_right.quiver(
                        pos[0], pos[1], pos[2],
                        error_vec[0], error_vec[1], error_vec[2],
                        length=error_norm * error_scale_factor,
                        normalize=False,
                        color='m',
                    )
                # --- Draw Force Vector ---
                if self.latest_force is not None:
                    # Remove previous 3D quiver
                    if self.force_quiver_3d:
                        self.force_quiver_3d.remove()
                        self.force_quiver_3d = None
                    # Remove previous 2D quivers
                    if self.force_quiver_top: self.force_quiver_top.remove(); self.force_quiver_top = None
                    if self.force_quiver_front: self.force_quiver_front.remove(); self.force_quiver_front = None
                    if self.force_quiver_right: self.force_quiver_right.remove(); self.force_quiver_right = None

                    # Draw new quiver at the latest EE position
                    pos = latest_ee_pos
                    force = self.latest_force
                    force_norm = np.linalg.norm(force)
                    scale_factor = 0.0005 # Adjust scale for visibility - Increased from 0.005

                    if force_norm > 1e-3: # Only draw if force is non-negligible
                         # Check if legend entry exists (3D)
                        legend_exists_3d = any(artist.get_label() == 'External Force' for artist in self.ax_3d.collections + self.ax_3d.lines)
                        # 3D Force Quiver
                        self.force_quiver_3d = self.ax_3d.quiver(
                            pos[0], pos[1], pos[2], # Start point
                            force[0], force[1], force[2], # Vector components
                            length=force_norm * scale_factor, # Scaled length
                            normalize=True, # Use direction, length controls magnitude display
                            color='g', # Green color for force
                            label='External Force' if not legend_exists_3d else "" # Add legend entry only once
                        )
                        self.force_quiver_top = self.ax_top.quiver(
                            pos[0], pos[1], pos[2],
                            force[0], force[1], force[2],
                            length=force_norm * scale_factor,
                            normalize=False,
                            color='g',
                        )   
                        self.force_quiver_front = self.ax_front.quiver(
                            pos[0], pos[1], pos[2],
                            force[0], force[1], force[2],
                            length=force_norm * scale_factor,
                            normalize=False,
                            color='g',
                        )
                        self.force_quiver_right = self.ax_right.quiver(
                            pos[0], pos[1], pos[2],
                            force[0], force[1], force[2],
                            length=force_norm * scale_factor,
                            normalize=False,
                            color='g',
                        )
                        
                         # Update legend if the quiver was added with a label (3D only for now)
                        if (not legend_exists_3d and self.force_quiver_3d) or \
                           (not error_legend_exists_3d and self.error_quiver_3d): # Update legend if either new quiver was added
                            self.ax_3d.legend()

        # Redraw the figure
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            # Ignore errors if the plot window is closed by the user
            if 'FigureCanvas' not in str(type(e)):
                 self.get_logger().error(f"Error updating plot: {e}")

    def destroy_node(self):
        plt.close(self.fig) # Close the plot window
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    plotter_node = PlotterNode()
    try:
        rclpy.spin(plotter_node)
    except KeyboardInterrupt:
        plotter_node.get_logger().info("Shutting down Plotter...")
    finally:
        plotter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 