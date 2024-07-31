import numpy as np
from tqdm import tqdm
from motion.motion_estimation import *
from utils.target import *
from algorithms.NUCnlFilter import M_n

def morgan(frames: list | np.ndarray, alpha=0.01):
    """
    Apply the Morgan algorithm to a sequence of frames for non-uniformity correction (NUC).

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan algo processing", unit="frame"):
        frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_frame(frame, img_nuc=0, alpha=0.01):
    """
    Perform SBNUCIRFPA on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    frame_est = frame + 2**13 - img_nuc

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est).astype(frame.dtype), img_nuc.astype(frame.dtype)

def morgan_moving(frames: list | np.ndarray, alpha=0.01, moving_rate=0.2):
    """
    Apply the Morgan algorithm with motion estimation to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.
        moving_rate (float, optional): Rate of the image change to consider movement. Defaults to 0.2.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    frame_thr = frames[0].shape[0] * frames[0].shape[1] * moving_rate
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame
    frame_est_n_1, img_nuc = morgan_frame(frames[0], img_nuc, alpha)
    img_nuc_n_1 = img_nuc 

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan moving algo processing", unit="frame"):
        frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
        # Perform motion estimation to decide if the frame should be corrected
        if M_n(frame=frame_est, frame_n_1=frame_est_n_1, Tg=frame_thr):  # motion estimation might be better with estimation frame comparison
            # update variables
            img_nuc_n_1 = img_nuc
            frame_est_n_1 = frame_est
            # add estimated frame to the rest
            all_frame_est.append(frame_est)

        else:
            # update variables
            img_nuc = img_nuc_n_1
            frame_est_n_1 = frame - img_nuc_n_1
            # add estimated frame to the rest
            all_frame_est.append(frame_est_n_1)
        
    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt(frames: list | np.ndarray, alpha=0.01):
    """
    Apply the Morgan algorithm with additional filtering to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_frame(frame, img_nuc, alpha=0.01):
    """
    Perform SBNUCIRFPA with additional filtering on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame with additional filtering
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    img_nuc_bas = frame_gauss_3x3_filtering(img_nuc)
    img_nuc_haut = frame_laplacian_3x3_filtering(img_nuc)
    frame_est = frame + 2**13 - (img_nuc_bas + img_nuc_haut)

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def morgan_filt_haut(frames: list | np.ndarray, alpha=0.01):
    """
    Apply the Morgan algorithm with high-pass filtering to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = frames[0] - frame_mean_filtering(frames[0], 5)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering haut algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_haut_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_haut_frame(frame, img_nuc, alpha=0.01):
    """
    Perform SBNUCIRFPA with high-pass filtering on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame with high-pass filtering
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    img_nuc_haut = img_nuc - frame_mean_filtering(img_nuc, 5)
    frame_est = frame + 2**13 - img_nuc_haut

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def Adamorgan(frames: list | np.ndarray, K=2**-3, A=2**-8):
    """
    Apply the Adamorgan algorithm to a sequence of frames for non-uniformity correction (NUC).

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        K (float, optional): Learning rate parameter. Defaults to 0.01.
        A (float, optional): Parameter for adaptive learning rate. Defaults to 0.2.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="Adamorgan algo processing", unit="frame"):
        frame_est, img_nuc = Adamorgan_frame(frame, img_nuc, K=K, A=A)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def Adamorgan_frame(frame, img_nuc=0, K=2**-3, A=2**-8):
    """
    Perform SBNUCIRFPA on a single frame using the Adamorgan algorithm.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        K (float): Learning rate parameter.
        A (float): Parameter for adaptive learning rate.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Calculate the adaptive learning rate
    eta = K / (1 + A * frame_var_filtering(frame)**2)
    # breakpoint()

    # Perform Morgan algorithm on the frame with adaptive learning rate
    img_nuc = eta * frame + (1 - eta) * img_nuc
    frame_est = frame + 2**13 - img_nuc

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est).astype(frame.dtype), img_nuc.astype(frame.dtype)
