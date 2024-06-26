# SBNUC algorithm based on interframe registration

## good sumary of all previous work

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *

def SBNUCif_reg(frames, algo='FourierShift', lr=0.05, offset_only=True):
    """
    Apply the Scene-Based Non-Uniformity Correction (SBNUC) algorithm with interframe registration to a sequence of frames.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        algo (str, optional): The motion estimation algorithm to use. Defaults to 'FourierShift'.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        np.ndarray: A numpy array of corrected frames.
    """
    all_frames_est = []
    frame_n_1 = frames[0]  # Initialize with the first frame

    coeffs = init_nuc(frame_n_1)  # Initialize NUC coefficients

    # Iterate through the frames starting from the second frame
    for frame in tqdm(frames[1:], desc="CstStatSBNUC processing", unit="frame"):
        frame_est, coeffs = SBNUCif_reg_frame(frame, frame_n_1, coeffs, lr, algo, offset_only)
        all_frames_est.append(frame_est)
        frame_n_1 = frame  # Update the previous frame for motion detection
    
    return np.array(all_frames_est, dtype=frames[0].dtype)

def SBNUCif_reg_frame(frame, frame_n_1, coeffs, lr=0.05, algo='FourierShift', offset_only=True):
    """
    Apply the SBNUC method to a single frame with interframe registration.

    Args:
        frame (np.ndarray): The current frame to be corrected.
        frame_n_1 (np.ndarray): The previous frame for motion estimation.
        coeffs (dict): The coefficients used for the correction.
        lr (float, optional): The learning rate for updating coefficients. Defaults to 0.1.
        algo (str, optional): The motion estimation algorithm to use. Defaults to 'FourierShift'.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    # Estimate the motion vector between the previous frame and the current frame
    di, dj = motion_estimation_frame(prev_frame=frame_n_1, curr_frame=frame, algo=algo)
    
    all_Xest = []  # List to store estimated pixel values for the frame
    
    # i, j position of current frame is equal to i-di, j-dj position of previous frame
    k=0
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            idi = i-di
            jdj = j-dj
            if (0<=idi<len(frame)) and (0<=jdj<len(frame[0])):
                Eij = frame_n_1[idi][jdj] - frame[i][j]
                coeffs['o'][i][j] = coeffs['o'][i][j] + lr * Eij

                if not offset_only:
                    coeffs['g'][i][j] = coeffs['g'][i][j] + lr * Eij * frame[i][j]

            # Estimate corrected pixel value
            all_Xest[k].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
        k+=1
    return np.array(all_Xest, dtype=frame.dtype), coeffs

def AdaSBNUCif_reg(frames, algo='FourierShift', lr=0.05, offset_only=True):
    """
    Apply the Adaptive Scene-Based Non-Uniformity Correction (AdaSBNUC) algorithm with interframe registration to a sequence of frames.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        algo (str, optional): The motion estimation algorithm to use. Defaults to 'FourierShift'.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        np.ndarray: A numpy array of corrected frames.
    """
    all_frames_est = []
    frame_n_1 = frames[0]  # Initialize with the first frame

    coeffs = init_nuc(frame_n_1)  # Initialize NUC coefficients

    # Iterate through the frames starting from the second frame
    for frame in tqdm(frames[1:], desc="CstStatSBNUC processing", unit="frame"):
        frame_est, coeffs = AdaSBNUCif_reg_frame(frame, frame_n_1, coeffs, lr, algo, offset_only)
        all_frames_est.append(frame_est)
        frame_n_1 = frame  # Update the previous frame for motion detection
    
    return np.array(all_frames_est, dtype=frames[0].dtype)

def AdaSBNUCif_reg_frame(frame, frame_n_1, coeffs, lr=0.05, algo='FourierShift', offset_only=True):
    """
    Apply the AdaSBNUC method to a single frame with interframe registration.
    Update the nuc coefficients only if motion distance is between 2 and 16.

    Args:
        frame (np.ndarray): The current frame to be corrected.
        frame_n_1 (np.ndarray): The previous frame for motion estimation.
        coeffs (dict): The coefficients used for the correction.
        lr (float, optional): The learning rate for updating coefficients. Defaults to 0.1.
        algo (str, optional): The motion estimation algorithm to use. Defaults to 'FourierShift'.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    # Estimate the motion vector between the previous frame and the current frame
    di, dj = motion_estimation_frame(prev_frame=frame_n_1, curr_frame=frame, algo=algo)
    
    all_Xest = []  # List to store estimated pixel values for the frame
    update_nuc = (2 <= np.sqrt(di**2 + dj**2) <= 16)  # Update threshold from the paper
    
    # i, j position of current frame is equal to i-di, j-dj position of previous frame
    k=0
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            idi = i-di
            jdj = j-dj
            if update_nuc and (0<=idi<len(frame)) and (0<=jdj<len(frame[0])):
                Eij = frame_n_1[idi][jdj] - frame[i][j]
                coeffs['o'][i][j] = coeffs['o'][i][j] + lr * Eij

                if not offset_only:
                    coeffs['g'][i][j] = coeffs['g'][i][j] + lr * Eij * frame[i][j]

            # Estimate corrected pixel value
            all_Xest[k].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
        k+=1

    return np.array(all_Xest, dtype=frame.dtype), coeffs
