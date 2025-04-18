import numpy as np


def figure_8(x_amplitude, z_amplitude, offset, timestep, period, num_periods, angle_offset=np.pi/4):
    x = lambda t: offset[0] + x_amplitude * np.sin(t)  # [-x_amplitude, x_amplitude]
    y = lambda t: offset[1]
    z = lambda t: offset[2] + z_amplitude * np.sin(2*t)/2 + z_amplitude/2  # [-z_amplitude/2, z_amplitude/2]
    
    # rotate around z-axis
    R = np.array([[np.cos(angle_offset), -np.sin(angle_offset), 0.0],
                  [np.sin(angle_offset), np.cos(angle_offset), 0.0],
                  [0.0, 0.0, 1.0]])
    
    def get_rotated_coords(t):
        unrot = np.array([x(t), y(t), z(t)])
        rot = R @ unrot
        return rot[0], rot[1], rot[2]
    
    # rotation coordinate functions
    x_rot = lambda t: get_rotated_coords(t)[0]
    y_rot = lambda t: get_rotated_coords(t)[1] 
    z_rot = lambda t: get_rotated_coords(t)[2]
    
    timesteps = np.linspace(0, 2*np.pi, int(period/timestep))
    
    fig_8 = np.array([[x_rot(t), y_rot(t), z_rot(t), 0.0, 0.0, 0.0] for t in timesteps]).flatten()
    fig_8 = np.tile(fig_8, num_periods)
    
    return fig_8