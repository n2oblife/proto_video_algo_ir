# Robust approach for NUC in IR FPA
# RobustNUCIRFPA

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *

def RobustNUCIRFPA(
        frames: list | np.ndarray, 
        alpha=0.01,
        ada=True,
        algo='FourierShift',
        offset_only=True
    ):
    """
    Apply the RobustNUCIRFPA algorithm to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        offset_only (bool, optional): If True, only correct the offset. Defaults to True.

    Returns:
        np.ndarray: Estimated frames after applying RobustNUCIRFPA.
    """
    all_frames_est = []
    coeffs = init_nuc(frames[0])  # Initialize NUC coefficients
    frame_n_1 = frames[0]
    for frame in tqdm(frames[1:], desc="RobustNUCIRFPA algorithm", unit="frame"):
        # Estimate the current frame using the previous frame
        if offset_only:
            frame_est, _, coeffs['o'] = RobustNUCIRFPA_frame(
                frame=frame,
                coeffs=coeffs,
                alpha=alpha,
                ada=ada,
                frame_n_1=frame_n_1,
                algo=algo
            )
        else:
            frame_est, coeffs['g'], coeffs['o'] = RobustNUCIRFPA_frame(
                frame=frame,
                coeffs=coeffs,
                alpha=alpha,
                ada=ada,
                frame_n_1=frame_n_1,
                algo=algo
            )
        all_frames_est.append(frame_est)
        frame_n_1 = frame
    return np.array(all_frames_est, dtype=frames.dtype)

def RobustNUCIRFPA_frame(
        frame: list | np.ndarray, 
        coeffs, 
        alpha = 0.01,
        ada=True, 
        frame_n_1=None, 
        algo='FourierShift'
    ) -> np.ndarray:
    """
    Process a single frame using the RobustNUCIRFPA algorithm.

    Args:
        frame (np.ndarray): Input frame.
        coeffs (dict): NUC coefficients.
        ada (bool, optional): If True, apply adaptive correction. Defaults to True.
        frame_n_1 (np.ndarray, optional): Previous frame. Defaults to None.
        algo (str, optional): Motion estimation algorithm. Defaults to 'FourierShift'.

    Returns:
        np.ndarray: Estimated frame after applying RobustNUCIRFPA.
    """
    if ada:
        # Apply Gaussian filtering for adaptive correction
        T = frame_gauss_3x3_filtering(frame)
        X_est = Xest(
            g=coeffs['g'],
            y=frame,
            o=coeffs['o'],
            b=0
        )
    else:
        # Apply motion estimation for non-adaptive correction
        dx, dy = motion_estimation_frame(frame, frame_n_1, algo)
        T = frame[max(0, dx):min(len(frame), dx)][max(0, dy):min(len(frame[0]), dy)]
        X_est = Xest(
            g=coeffs['g'],
            y=frame[max(0, dx):min(len(frame), dx)][max(0, dy):min(len(frame[0]), dy)],
            o=coeffs['o']
        )
    Eij = X_est - T
    return X_est, sgd_step(coeffs['g'], alpha, Eij * frame), sgd_step(coeffs['o'], alpha, Eij)

def eij(Eij, mu, sig, k=3):
    """
    Compute the error term for robust estimation.

    Args:
        Eij (np.ndarray): Error image.
        mu (float): Mean of the error image.
        sig (float): Standard deviation of the error image.
        k (int, optional): Threshold factor. Defaults to 3.

    Returns:
        np.ndarray: Error term for robust estimation.
    """
    if np.abs(Eij - mu) >= k * sig:
        return 0
    else:
        return Eij
    
def apply_eij_elementwise(error_image, k=3):
    """
    Apply the eij function element-wise to an error image.

    Args:
        error_image (np.ndarray): The error image to be filtered.
        k (int, optional): Threshold factor. Defaults to 3.

    Returns:
        np.ndarray: Filtered error image.
    """
    mu = np.mean(error_image)
    sig = np.std(error_image)
    return np.vectorize(lambda x: eij(x, mu, sig, k))(error_image)

def AdaRobustNUCIRFPA(
        frames: list | np.ndarray, 
        alpha_m=0.05,
        ada=True,
        algo='FourierShift',
        offset_only=True, 
    ):
    """
    Apply the Adaptive RobustNUCIRFPA algorithm to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        offset_only (bool, optional): If True, only correct the offset. Defaults to True.

    Returns:
        np.ndarray: Estimated frames after applying Adaptive RobustNUCIRFPA.
    """
    all_frames_est = []
    frame_n_1 = frames[0]
    coeffs = init_nuc(frames[0])  # Initialize NUC coefficients
    for frame in tqdm(frames[1:], desc="Adaptive RobustNUCIRFPA algorithm", unit="frame"):
        # Estimate the current frame using the previous frame
        if offset_only:
            frame_est, _, coeffs['o'] = AdaRobustNUCIRFPA_frame(
                frame=frame,
                coeffs=coeffs,
                ada=ada,
                frame_n_1=frame_n_1,
                algo=algo,
                alpha_m=alpha_m
            )
        else:
            frame_est, coeffs['g'], coeffs['o'] = AdaRobustNUCIRFPA_frame(
                frame=frame,
                coeffs=coeffs,
                ada=ada,
                frame_n_1=frame_n_1,
                algo=algo,
                alpha_m=alpha_m
            )
        all_frames_est.append(frame_est)
        frame_n_1 = frame
    return np.array(all_frames_est, dtype=frames.dtype)

def AdaRobustNUCIRFPA_frame(
        frame: list | np.ndarray, 
        coeffs, 
        ada=True, 
        frame_n_1=None, 
        algo='FourierShift',
        alpha_m=0.05
    ) -> np.ndarray:
    """
    Process a single frame using the Adaptive RobustNUCIRFPA algorithm.

    Args:
        frame (np.ndarray): Input frame.
        coeffs (dict): NUC coefficients.
        ada (bool, optional): If True, apply adaptive correction. Defaults to True.
        frame_n_1 (np.ndarray, optional): Previous frame. Defaults to None.
        algo (str, optional): Motion estimation algorithm. Defaults to 'FourierShift'.
        alpha_m (float, optional): Maximum learning rate. Defaults to 0.2.
        E_m (float, optional): Initial error estimate. Defaults to 1.

    Returns:
        np.ndarray: Estimated frame after applying Adaptive RobustNUCIRFPA.
    """
    if ada:
        # Apply Gaussian filtering for adaptive correction
        T = frame_gauss_3x3_filtering(frame)
        X_est = Xest(
            g=coeffs['g'],
            y=frame,
            o=coeffs['o'],
            b=0
        )
    else:
        # Apply motion estimation for non-adaptive correction
        dx, dy = motion_estimation_frame(frame, frame_n_1, algo)
        T = frame[max(0, dx):min(len(frame), dx)][max(0, dy):min(len(frame[0]), dy)]
        X_est = Xest(
            g=coeffs['g'],
            y=frame[max(0, dx):min(len(frame), dx)][max(0, dy):min(len(frame[0]), dy)],
            o=coeffs['o']
        )
    E = X_est - T
    Eij = apply_eij_elementwise(E)
    E_m = np.mean((X_est - frame_mean_filtering(frame))**2)
    alpha = alpha_n(Eij, E_m, alpha_m)
    return X_est, sgd_step(coeffs['g'], alpha, Eij * frame), sgd_step(coeffs['o'], alpha, Eij)

def alpha_n(Eij, E_m, alpha_m=0.05, C=5, reg=True)->float:
    """
    Compute the adaptive learning rate.

    Args:
        Eij (np.ndarray): Error term.
        alpha_m (float): Maximum learning rate.
        E_m (float): Initial error estimate.
        C (int, optional): Regularization factor. Defaults to 5.
        reg (bool, optional): If True, apply regularization. Defaults to True.

    Returns:
        float: Adaptive learning rate.
    """
    if reg:
        return alpha_m * E_m / (np.mean(Eij * Eij))
    else:
        return C * (np.mean(Eij * Eij))
