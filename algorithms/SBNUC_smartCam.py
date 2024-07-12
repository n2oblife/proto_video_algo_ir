# SBNUC from algorithm to implementing a smart camera
# SBNUC_smartCam

# This implementation focuses on hardware-oriented pipelines with different processing stages.
# The last pipeline is implemented but not tested. It includes offset-only corrections.
# TODO: Check if the low pass and high pass filters are correct or need adjustments.
# Refer to: http://ressources.unit.eu/cours/videocommunication/UNIT_Image%20Processing_nantes/Version%20US/Chapter%203/Courses/Linear%20Filtering/Rchap3_linearFiltering_US[final].pdf

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *

def SBNUC_smartCam_pipeA(frames:list|np.ndarray, alpha=2**(-8)):
    """
    Apply the SBNUC_smartCam algorithm pipeline A to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        alpha (float, optional): Alpha parameter for exponential window filtering. Defaults to 2**(-8).

    Returns:
        np.ndarray: Estimated frames after applying pipeline A.
    """
    all_frames_est = []
    m_k = np.zeros(frames[0].shape, dtype=frames.dtype)
    for frame in tqdm(frames, desc="SBNUC_smartCam algorithm pipeline A", unit="frame"):
        frame_est, m_k = SBNUC_smartCam_pipeA_frame(frame, m_k, alpha)
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)

def SBNUC_smartCam_pipeA_frame(frame:list|np.ndarray, m_k, alpha=2**(-8)):
    """
    Process a single frame using pipeline A.

    Args:
        frame (np.ndarray): Input frame.
        m_k (np.ndarray): Previous correction map.
        alpha (float, optional): Alpha parameter for exponential window filtering. Defaults to 2**(-8).

    Returns:
        tuple: (Estimated frame, updated correction map)
    """
    high_passed = frame_military_3x3_filtering(frame)
    # high_passed = frame_sobel_3x3_filtering(frame)
    m_k = frame_exp_window_filtering(high_passed, m_k, alpha)
    return SBNUC_smartCam_apply_corr(frame, m_k), m_k

def SBNUC_smartCam_pipeB(frames:list|np.ndarray, alpha=2**(-8), alpha_p=2**(-12)):
    """
    Apply the SBNUC_smartCam algorithm pipeline B to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        np.ndarray: Estimated frames after applying pipeline B.
    """
    all_frames_est = []
    m_k = np.zeros(frames[0].shape, dtype=frames.dtype)
    m_k_p = np.zeros(len(frames[0]), dtype=frames.dtype)
    for frame in tqdm(frames, desc="SBNUC_smartCam algorithm pipeline B", unit="frame"):
        frame_est, m_k, m_k_p = SBNUC_smartCam_pipeB_frame(frame, m_k, m_k_p, alpha, alpha_p)
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)

def SBNUC_smartCam_pipeB_frame(
        frame:list|np.ndarray, 
        m_k:list|np.ndarray, 
        m_k_p:list|np.ndarray, 
        alpha=2**(-8), alpha_p=2**(-12)
    )->np.ndarray:
    """
    Process a single frame using pipeline B.

    Args:
        frame (np.ndarray): Input frame.
        m_k (np.ndarray): Previous pixel-level correction map.
        m_k_p (np.ndarray): Previous column-level correction map.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        tuple: (Estimated frame, updated pixel-level correction map, updated column-level correction map)
    """
    high_passed = frame_military_3x3_filtering(frame)
    # high_passed = frame_sobel_3x3_filtering(frame)
    m_k_p = SBNUC_smartCam_col_corr(frame, m_k_p, alpha_p)
    m_k = frame_exp_window_filtering(high_passed, m_k, alpha)
    return SBNUC_smartCam_apply_corr(SBNUC_smartCam_apply_col_corr(frame, m_k_p), m_k), m_k, m_k_p

def SBNUC_smartCam_pipeC(frames:list|np.ndarray, alpha=2**(-8), alpha_p=2**(-12), alpha_avg=0.01):
    """
    Apply the SBNUC_smartCam algorithm pipeline C to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).
        alpha_avg (float, optional): Alpha parameter for frame averaging. Defaults to 0.1.

    Returns:
        np.ndarray: Estimated frames after applying pipeline C.
    """
    all_frames_est = []
    frame_avg = np.zeros(frames[0].shape, dtype=frames.dtype)
    m_k = np.zeros(frames[0].shape, dtype=frames.dtype)
    m_k_p = np.zeros(len(frames[0]), dtype=frames.dtype)
    for frame in tqdm(frames, desc="SBNUC_smartCam algorithm pipeline C", unit="frame"):
        # m_k_p = np.zeros(len(frames[0]), dtype=frames.dtype)
        frame_est, m_k, m_k_p = SBNUC_smartCam_pipeC_frame(frame, frame_avg, m_k, m_k_p, alpha, alpha_p, alpha_avg)
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)

def SBNUC_smartCam_pipeC_frame(
        frame:list|np.ndarray, 
        frame_avg:list|np.ndarray, 
        m_k:list|np.ndarray, 
        m_k_p:list|np.ndarray, 
        alpha=2**(-8), alpha_p=2**(-12), alpha_avg=0.01
    )->np.ndarray:
    """
    Process a single frame using pipeline C.

    Args:
        frame (np.ndarray): Input frame.
        frame_avg (np.ndarray): Averaged frame for background estimation.
        m_k (np.ndarray): Previous pixel-level correction map.
        m_k_p (np.ndarray): Previous column-level correction map.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).
        alpha_avg (float, optional): Alpha parameter for frame averaging. Defaults to 0.1.

    Returns:
        tuple: (Estimated frame, updated pixel-level correction map, updated column-level correction map)
    """
    piped_im = np.zeros(frame.shape, dtype=frame.dtype)
    frame_avg = frame_exp_window_filtering(frame, frame_avg, alpha_avg)
    m_k_p = SBNUC_smartCam_col_corr(frame_avg, m_k_p, alpha_p)
    piped_im = SBNUC_smartCam_apply_col_corr(frame, m_k_p)
    frame_est, m_k = SBNUC_smartCam_pipeA_frame(piped_im, m_k, alpha)
    return frame_est, m_k, m_k_p

def SBNUC_smartCam_col_corr(frame:list|np.ndarray, m_k_p:list|np.ndarray, alpha_p=2**(-12)):
    """
    Apply column-level correction using exponential window filtering.

    Args:
        frame (np.ndarray): Input frame.
        m_k_p (np.ndarray): Previous column-level correction map.
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        np.ndarray: Updated column-level correction map.
    """
    if isinstance(frame, list):
        for j in range(len(frame[0])):
            for i in range(len(frame)):
                m_k_p[i] = exp_window(frame[i][j], m_k_p[i], alpha_p)
        return m_k_p
    elif isinstance(frame, np.ndarray):
        return (1 - alpha_p) * m_k_p + alpha_p * np.mean(frame.T, axis=0)
    else:
        raise NotImplementedError

def SBNUC_smartCam_row_corr(frame:list|np.ndarray, m_k_r:list|np.ndarray, alpha_p=2**(-12)):
    """
    Apply column-level correction using exponential window filtering.

    Args:
        frame (np.ndarray): Input frame.
        m_k_p (np.ndarray): Previous column-level correction map.
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        np.ndarray: Updated column-level correction map.
    """
    if isinstance(frame, list):
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                m_k_r[j] = exp_window(frame[i][j], m_k_r[j], alpha_p)
        return m_k_r
    elif isinstance(frame, np.ndarray):
        return (1 - alpha_p) * m_k_r + alpha_p * np.mean(frame, axis=1)
    else:
        raise NotImplementedError

def SBNUC_smartCam_apply_corr(frame:list|np.ndarray, m_k_p:list|np.ndarray):
    """
    Apply the column-level correction map to the input frame.

    Args:
        frame (np.ndarray): Input frame.
        m_k_p (np.ndarray): Column-level correction map.

    Returns:
        np.ndarray: Corrected frame.
    """
    if isinstance(frame, list):
        corrected_im = np.array(frame, dtype=frame.dtype)
        for j in range(len(frame[0])):
            for i in range(len(frame)):
                corrected_im[i][j] = frame[i][j] - m_k_p[i]
        return corrected_im
    elif isinstance(frame, np.ndarray):
        corrected_im = frame - m_k_p + 2**13
        return np.where(corrected_im < 0, 0, corrected_im)

def SBNUC_smartCam_apply_col_corr(frame:list|np.ndarray, m_k_p:list|np.ndarray):
    """
    Apply the column-level correction map to the input frame.

    Args:
        frame (np.ndarray): Input frame.
        m_k_p (np.ndarray): Column-level correction map.

    Returns:
        np.ndarray: Corrected frame.
    """
    if isinstance(frame, list):
        corrected_im = np.array(frame, dtype=frame.dtype)
        for j in range(len(frame[0])):
            for i in range(len(frame)):
                corrected_im[i][j] = frame[i][j] - m_k_p[i]
        return corrected_im
    elif isinstance(frame, np.ndarray):
        corrected_im = frame - m_k_p[:, np.newaxis] + 2**13
        return np.where(corrected_im < 0, 0, corrected_im)


def SBNUC_smartCam_own_pipe(frames:list|np.ndarray, alpha=2**(-8), alpha_p=2**(-12)):
    """
    Apply the SBNUC_smartCam algorithm pipeline B to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        np.ndarray: Estimated frames after applying pipeline B.
    """
    all_frames_est = []
    m_k = np.zeros(frames[0].shape, dtype=frames.dtype)
    m_k_p = np.zeros(len(frames[0]), dtype=frames.dtype)
    m_k_r = np.zeros(len(frames[0]), dtype=frames.dtype)
    for frame in tqdm(frames, desc="SBNUC_smartCam algorithm own pipeline", unit="frame"):
        frame_est, m_k, m_k_p, m_k_r = SBNUC_smartCam_own_pipe_frame(
            frame, m_k, m_k_p, m_k_r, alpha, alpha_p
            )
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)

def SBNUC_smartCam_own_pipe_frame(
        frame:list|np.ndarray, 
        m_k:list|np.ndarray, 
        m_k_p:list|np.ndarray, 
        m_k_r:list|np.ndarray, 
        alpha=2**(-8), alpha_p=2**(-12)
    )->np.ndarray:
    """
    Process a single frame using pipeline B.

    Args:
        frame (np.ndarray): Input frame.
        m_k (np.ndarray): Previous pixel-level correction map.
        m_k_p (np.ndarray): Previous column-level correction map.
        alpha (float, optional): Alpha parameter for pixel-level correction. Defaults to 2**(-8).
        alpha_p (float, optional): Alpha parameter for column-level correction. Defaults to 2**(-12).

    Returns:
        tuple: (Estimated frame, updated pixel-level correction map, updated column-level correction map)
    """
    high_passed = frame_military_3x3_filtering(frame)
    # high_passed = frame_sobel_3x3_filtering(frame)
    m_k_p = SBNUC_smartCam_col_corr(frame, m_k_p, alpha_p)
    m_k_r = SBNUC_smartCam_row_corr(frame, m_k_r, alpha_p)
    m_k = frame_exp_window_filtering(high_passed, m_k, alpha)
    return SBNUC_smartCam_apply_col_corr(frame, m_k_p+m_k_r) - m_k, m_k, m_k_p, m_k_r