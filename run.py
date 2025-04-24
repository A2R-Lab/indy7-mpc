import numpy as np
from datetime import datetime
import torch
import sys 
import os
sys.path.append('./src')
import time
import rclpy
from rclpy.node import Node

from reference_traj import figure_8
from gato_node import GATO_Node


def main(args=None):
    np.random.seed(42)
    try:
        rclpy.init(args=args)
        dt = 0.02
        N = 32
        
        # # ----- Single solver -----
        
        # gato_node = GATO_Node(ref_traj=ref_traj, batch_size=1, N=N, dt=dt,
        #                                   f_ext_std=0.0, f_ext_resample_std=0.0, 
        #                                   f_ext_actual=f_ext_actual)
        
        # ----- 32 BATCH -----

        # gato_node = GATO_Node(ref_traj=ref_traj, batch_size=32, N=N, dt=dt,
        #                                   f_ext_std=2.0, f_ext_resample_std=0.2, 
        #                                   f_ext_actual=f_ext_actual)
        
        # ----- 64 BATCH -----

        gato_node = GATO_Node(batch_size=64, N=N, dt=dt,
                                          f_ext_std=10.0, f_ext_resample_std=0.2)
        
        # ---------------
        
        rclpy.spin(gato_node)
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()