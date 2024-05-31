# SBNUC from algorithm to implementing a smart camera
# SBNUC_smartCam

# very hardware oriented and differents pipelines (implement last pipeline but don't test)

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *

def SBNUC_smartCam(frames, offset_only:True):
    all_frames_est = []
    coeffs = init_nuc(frames[0])  # Initialize NUC coefficients

    return