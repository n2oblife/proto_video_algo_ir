# SBNUC using complementary fixed pattern noise models
# SBNUCcomplement

# This paper could be used as a complement to the first filter

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *

def SBNUCcomplement(frames:list | np.ndarray):
    """
    Apply the SBNUCcomplement algorithm to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.

    Returns:
        np.ndarray: Estimated frames after applying the SBNUCcomplement algorithm.
    """
    all_frames_est = []
    Cn = frames[0]  # Cumulative mean image
    frame_n_1 = frames[0]  # Previous frame
    for frame in tqdm(frames[1:], desc="SBNUCcomplement algorithm", unit="frame"):
        frame_est, Cn = SBNUCcomplement_frame(frame, frame_n_1, Cn)
        frame_n_1 = frame  # Update the previous frame
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames.dtype)

def SBNUCcomplement_frame(frame:list | np.ndarray, frame_n_1:list | np.ndarray, Cn:list | np.ndarray):
    """
    Process a single frame using the SBNUCcomplement algorithm.

    Args:
        frame (np.ndarray): Current frame.
        frame_n_1 (np.ndarray): Previous frame.
        Cn (np.ndarray): Cumulative mean image.

    Returns:
        tuple: (Estimated frame, updated cumulative mean image)
    """
    wn = w_n(frame, frame_n_1)  # Compute the weight for the current frame
    Mx = frame_gauss_3x3_filtering(frame)  # Apply Gaussian filtering to the current frame
    Cn = exp_window(Cn, Mx, wn)  # Update the cumulative mean image
    return frame - (Cn - np.mean(Cn)), Cn  # Return the estimated frame and updated cumulative mean image

def w_n(frame:list | np.ndarray, frame_n_1:list | np.ndarray, threshold=2):
    """
    Compute the weight for the current frame based on scene detail and motion.

    Args:
        frame (np.ndarray): Current frame.
        frame_n_1 (np.ndarray): Previous frame.
        threshold (float, optional): Threshold for motion estimation. Defaults to 2.

    Returns:
        float: Weight for the current frame.
    """
    c = 50  # Proportionality constant
    phi = np.mean(frame_sobel_3x3_filtering(frame))  # Scene detail magnitude
    dx, dy = motion_estimation_frame(frame, frame_n_1, algo='FourierShift')  # Motion estimation
    if np.abs(dx**2 + dy**2) > threshold:  # Check if motion exceeds threshold
        return phi / (phi + threshold / c)  # Compute the weight
    else:
        return 0  # If motion is below threshold, return zero weight
