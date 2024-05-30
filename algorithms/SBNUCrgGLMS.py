# SBNUC with reduced ghosting using gated LMS algo
# SBNUCrgGLMS
###close to AdaSBNUCIRFPA with updating depending on the camera motion
import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *


def init_S_M(
        frame: list | np.ndarray,
        k_size=3
    ) -> dict[str, np.ndarray]:
    """
    Initialize the statistics for SBNUC by calculating the mean and mean absolute deviation (MAD)
    of the given frame using a specified kernel size.

    Args:
        frame (list | np.ndarray): The input frame as a 2D list or array.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the initial estimates for 's' (mean) and 'm' (MAD).
    """
    return {
        "s": frame_mean_filtering(frame, k_size),   # Initialize the mean using frame mean filtering
        "m": frame_mad_filtering(frame, k_size)     # Initialize the MAD using frame MAD filtering
    }


def CstStatSBNUC(frames: list | np.ndarray, threshold=0., k_size=3, offset_only=True) -> np.ndarray:
    """
    Apply the scene-based non-uniformity correction (SBNUC) method to a sequence of frames.

    This function iterates through a sequence of frames and applies non-uniformity correction to each frame
    based on initial coefficients. The corrected frames are collected and returned as a list or numpy array.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        threshold (float, optional): Error threshold for updating coefficients. Defaults to 0.
        k_size (int, optional): The size of the kernel used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.
        
    Returns:
        np.ndarray: An array of corrected frames.
    """
    # Initialize variables
    all_frames_est = []
    frame_n_1 = frames[0]

    # Initialize correction coefficients and SBNUC coefficients
    corr_coeffs = init_nuc(frames[0])
    sbnuc_coeffs = init_S_M(frames[0])

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="CstStatSBNUC processing", unit="frame"):
        # Estimate the current frame using the previous frame
        frame_est, corr_coeffs, sbnuc_coeffs = CstStatSBNUC_frame(
            frame, frame_n_1, 
            corr_coeffs, sbnuc_coeffs, 
            threshold, k_size, offset_only
        )
        all_frames_est.append(frame_est)
        
        # Update the previous frame for motion detection
        frame_n_1 = frame
    
    print(f"{len(all_frames_est)} frames estimated using SBNUC algorithm")
    return np.array(all_frames_est, dtype=frames[0].dtype) 


def CstStatSBNUC_frame(
        frame: list | np.ndarray, 
        frame_n_1: list | np.ndarray, 
        corr_coeffs: dict[str, float],
        sbnuc_coeffs: dict[str, float],
        threshold=0.,
        alpha=0.5,
        offset_only=True
    ) -> tuple[list | np.ndarray, dict[str, float], dict[str, float]]:
    """
    Apply the SBNUC method to a single frame.

    This function processes a single frame using the provided coefficients and updates them.
    It returns the corrected frame along with the updated coefficients.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        frame_n_1 (list | np.ndarray): The previous frame.
        corr_coeffs (dict[str, float]): The coefficients used for the correction.
        sbnuc_coeffs (dict[str, float]): The SBNUC coefficients for mean and MAD.
        threshold (float, optional): Error threshold for updating coefficients. Defaults to 0.
        alpha (float, optional): Exponential weighting factor. Defaults to 0.5.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[list | np.ndarray, dict[str, float], dict[str, float]]: The corrected frame and updated coefficients.
    """
    all_Xest = []  # List to store estimated pixel values for the frame
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            if offset_only:
                # Update only the offset coefficient
                _, corr_coeffs['o'][i][j], sbnuc_coeffs['s'][i][j], sbnuc_coeffs['m'][i][j] = constant_statistics_sbunc_update_nuc(
                    frame[i][j], frame_n_1[i][j],
                    sbnuc_coeffs["s"][i][j], sbnuc_coeffs["m"][i][j],
                    threshold, alpha
                )
            else:
                # Update both gain and offset coefficients
                corr_coeffs['g'][i][j], corr_coeffs['o'][i][j], sbnuc_coeffs['s'][i][j], sbnuc_coeffs['m'][i][j] = constant_statistics_sbunc_update_nuc(
                    frame[i][j], frame_n_1[i][j],
                    sbnuc_coeffs["s"][i][j], sbnuc_coeffs["m"][i][j],
                    threshold, alpha
                )
            # Estimate corrected pixel value
            all_Xest[i].append(Xest(corr_coeffs["g"][i][j], frame[i][j], corr_coeffs["o"][i][j]))
    return np.array(all_Xest, dtype=frame.dtype), corr_coeffs, sbnuc_coeffs


def constant_statistics_sbunc_update_nuc(Yij, Yij_n_1, S_n_1, M_n_1, threshold=0, alpha=0.5):
    """
    Update the constant statistics used in Scene-Based Non-Uniformity Correction (SBNUC) algorithm.

    Args:
        Yij (float): Current pixel value.
        Yij_n_1 (float): Previous pixel value.
        S_n_1 (float): Previous estimate of scene-based non-uniformity standard deviation.
        M_n_1 (float): Previous estimate of scene-based non-uniformity mean.
        threshold (float, optional): Error threshold. Defaults to 0.
        alpha (float, optional): Exponential weighting factor. Defaults to 0.5.

    Returns:
        tuple[float, float, float, float]: Updated coefficients for the correction.

            -> 1 / S_n: Updated gain NUC correction coefficient.

            -> -M_n / S_n: Updated offset NUC correction coefficient.

            -> S_n: Updated standard deviation.
            
            -> M_n: Updated mean.
    """
    if compute_error(Yij, Yij_n_1) >= threshold:
        M_n = exp_window(M_n_1, Yij, alpha)
        S_n = exp_window(S_n_1, compute_error(Yij, M_n), alpha)
        return 1 / S_n, -M_n / S_n, S_n, M_n
    else:
        return 1 / S_n_1, -M_n_1 / S_n_1, S_n_1, M_n_1

def SBNUCLMS(frames: list | np.ndarray, K=0.1, M=0.5, threshold=0, k_size=3, offset_only=True) -> np.ndarray:
    """
    Apply the scene-based non-uniformity correction (SBNUC) method to a sequence of frames.

    This function iterates through a sequence of frames and applies non-uniformity correction to each frame
    based on initial coefficients. The corrected frames are collected and returned as a list or numpy array.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        K (float): Maximum step size for the update. Defaults to 0.1.
        M (float): Data normalization factor. Defaults to 0.5.
        threshold (float, optional): Error threshold for the update. Defaults to 0.
        k_size (int, optional): The size of the kernel used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        np.ndarray: A numpy array of corrected frames.
    """
    all_frames_est = []  # List to store the estimated (corrected) frames
    corr_coeffs = init_nuc(frames[0])  # Initialize the correction coefficients based on the first frame
    Z = np.full(frames[0].shape, np.inf, dtype=frames[0].dtype)  # Initialize the Z matrix with infinite values

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames, desc="SBNUCLMS processing", unit="frame"):
        # Estimate the current frame using SBNUCLMS_frame function
        frame_est, corr_coeffs = SBNUCLMS_frame(frame, corr_coeffs, Z, K, M, threshold, k_size, offset_only)
        all_frames_est.append(frame_est)  # Append the estimated frame to the list

    print(f"{len(all_frames_est)} frames estimated using SBNUCrgGLMS algo")
    return np.array(all_frames_est, dtype=frames[0].dtype)  # Convert the list of frames to a numpy array and return it

def SBNUCLMS_frame(frame: list | np.ndarray, corr_coeffs: dict, Z, K=0.1, M=0.5, threshold=0, k_size=3, offset_only=True) -> tuple[list | np.ndarray, dict]:
    """
    Apply the SBNUC method to a single frame.

    This function processes a single frame using the provided coefficients and updates them.
    It returns the corrected frame along with the updated coefficients.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        corr_coeffs (dict): The coefficients used for the correction.
        Z (np.ndarray): The matrix for storing intermediate values.
        K (float): Maximum step size for the update. Defaults to 0.1.
        M (float): Data normalization factor. Defaults to 0.5.
        threshold (float, optional): Error threshold for the update. Defaults to 0.
        k_size (int, optional): The size of the kernel used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[list | np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    all_Xest = []  # List to store estimated pixel values for the frame

    # Apply a Gaussian filter to smooth the frame
    B = gaussian_filter(frame, sigma=k_size)

    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            Eij = compute_error(B[i][j], Z[i][j])  # Compute the error between the filtered frame and Z
            if Eij >= threshold:
                Z[i][j] = B[i][j]  # Update Z with the current value if the error exceeds the threshold
                # Compute the learning rate for the current pixel using SBNUCLMS_eta
                eta = SBNUCLMS_eta(build_kernel(frame, i, j, k_size), K, M)
                # Update the offset coefficient using stochastic gradient descent
                corr_coeffs["o"][i][j] = sgd_step(corr_coeffs["o"][i][j], eta, Eij)
                if not offset_only:
                    # Update the gain coefficient if offset_only is False
                    corr_coeffs["g"][i][j] = sgd_step(corr_coeffs["g"][i][j], eta, Eij * frame[i][j])
            # Estimate corrected pixel value
            all_Xest[i].append(Xest(corr_coeffs["g"][i][j], frame[i][j], corr_coeffs["o"][i][j]))
    
    return np.array(all_Xest, dtype=frame.dtype), corr_coeffs  # Return the corrected frame and updated coefficients

def SBNUCLMS_eta(subframe, K, M) -> float:
    """
    Compute the learning rate for a given subframe in the SBNUC method.

    Args:
        subframe (list | np.ndarray): A subframe (kernel) extracted from the original frame.
        K (float): Maximum step size for the update.
        M (float): Data normalization factor.

    Returns:
        float: The computed learning rate.
    """
    # Compute the variance of the kernel and use it to calculate the learning rate
    return K / (1 + (M**2) * (kernel_var_filtering(subframe)**2))
