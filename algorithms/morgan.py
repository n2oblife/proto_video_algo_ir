import numpy as np
from tqdm import tqdm
from motion.motion_estimation import *
from utils.target import *
from algorithms.NUCnlFilter import M_n


def morgan(frames: list | np.ndarray, alpha = 0.01):
    """


    Args:
        frames (list | np.ndarray): 
        alpha (float, optional): . Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    all_frame_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)     # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan algo processing", unit="frame"):
        frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_frame(frame, img_nuc=0, alpha = 0.01):
    """
    Perform SBNUCIRFPA on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform morgan algorithm on the frame
    img_nuc = alpha*frame + (1-alpha) * img_nuc
    frame_est = frame + 2**13 - img_nuc

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def morgan_moving(frames: list | np.ndarray, alpha = 0.01, algo='FourierShift', threshold = 2):
    """


    Args:
        frames (list | np.ndarray): 
        alpha (float, optional): . Defaults to 0.01.


    Returns:
        _type_: _description_
    """
    all_frame_est = []      # List to store estimated (corrected) frames
    img_nuc = frames[0]     # Initialize coefficients based on the first frame
    frame_n_1 = frames[0]

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan moving algo processing", unit="frame"):
        # Estimate the motion vector between the previous frame and the current frame
        # di, dj = motion_estimation_frame(prev_frame=frame_n_1, curr_frame=frame, algo=algo)
        # if np.sqrt(di**2+dj**2) > threshold :
        if M_n(frame=frame, frame_n_1=frame_n_1) : # motion estimation
            frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
            all_frame_est.append(frame_est)
        else :
            all_frame_est.append(frame - img_nuc)
        
        frame_n_1 = frame

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt(frames: list | np.ndarray, alpha = 0.01):
    """


    Args:
        frames (list | np.ndarray): 
        alpha (float, optional): . Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    all_frame_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)     # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_frame(frame, img_nuc, alpha = 0.01):
    """
    Perform SBNUCIRFPA on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform morgan algorithm on the frame
    img_nuc = alpha*frame + (1-alpha) * img_nuc
    img_nuc_bas = frame_gauss_3x3_filtering(img_nuc)
    img_nuc_haut = frame_laplacian_3x3_filtering(img_nuc)
    frame_est = frame + 2**13 - (img_nuc_bas + img_nuc_haut)

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def morgan_filt_haut(frames: list | np.ndarray, alpha = 0.01):
    """


    Args:
        frames (list | np.ndarray): 
        alpha (float, optional): . Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    all_frame_est = []      # List to store estimated (corrected) frames
    # img_nuc = np.zeros(frames[0].shape)     # Initialize coefficients based on the first frame
    img_nuc = frames[0]-frame_mean_filtering(frames[0], 5)     # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering haut algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_haut_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_haut_frame(frame, img_nuc, alpha = 0.01):
    """
    Perform SBNUCIRFPA on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform morgan algorithm on the frame
    img_nuc = alpha*frame + (1-alpha) * img_nuc
    img_nuc_haut = img_nuc-frame_mean_filtering(img_nuc, 5)
    frame_est = frame + 2**13 - img_nuc_haut

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc