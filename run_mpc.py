import numpy as np
from datetime import datetime
import torch
import sys 
import os
sys.path.append('./src')
import time
import rclpy
from gato_node import GatoNode

def main(args=None):
    np.random.seed(42)
    try:
        rclpy.init(args=args)
        dt = 0.02
        N = 32
        
        gato_node = GatoNode(batch_size=64, N=N, dt=dt,
                                          f_ext_std=10.0, f_ext_resample_std=0.2)
    
        rclpy.spin(gato_node)
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        gato_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()