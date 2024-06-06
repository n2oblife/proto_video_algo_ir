# Adaptive scene-based non-uniformity correction method for infrared focal plane arrays
# AdaSBNUCIRFPA (paper)

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *

def SBNUCIRFPA(frames: list | np.ndarray, eta: float | np.ndarray = 0.01, k_size=3, offset_only=True) -> np.ndarray:
    """
    Apply the scene-based non-uniformity correction (SBNUC) method to a sequence of frames.

    This function iterates through a sequence of frames and applies non-uniformity correction to each frame
    based on initial coefficients. The corrected frames are collected and returned as a list or numpy array.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        eta (float | np.ndarray, optional): The learning rate for updating coefficients. Defaults to 0.01.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.
        
    Returns:
        list | np.ndarray: A list or array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    coeffs = init_nuc(frames[0])  # Initialize coefficients based on the first frame
    
    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames, desc="SBNUCIRFPA processing", unit="frame"):
        frame_est, coeffs = SBNUCIRFPA_frame(frame, coeffs, eta, k_size, offset_only)
        all_frame_est.append(frame_est)
    
    print(f"{len(all_frame_est)} frames estimated using SBNUCIRFPA algo")
    return np.array(all_frame_est, dtype=frames[0].dtype)

def SBNUCIRFPA_frame(frame: list | np.ndarray, coeffs: dict, eta: float | np.ndarray = 0.01, k_size=3, offset_only = True) -> tuple[list | np.ndarray, dict]:
    """
    Apply the SBNUC method to a single frame.

    This function processes a single frame using the provided coefficients and updates them.
    It returns the corrected frame along with the updated coefficients.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        coeffs (dict): The coefficients used for the correction.
        eta (float | np.ndarray, optional): The learning rate for updating coefficients. Defaults to 0.01.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[list | np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    all_Xest = []  # List to store estimated pixel values for the frame
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            if offset_only:
                # Update coefficients based on local kernel around the pixel
                _ , coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    build_kernel(frame, i, j, k_size), 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta
                )
            else:
                # Update coefficients based on local kernel around the pixel
                coeffs["g"][i][j], coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    build_kernel(frame, i, j, k_size), 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta
                )
            # Estimate corrected pixel value
            all_Xest[i].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
    return np.array(all_Xest, dtype=frame.dtype), coeffs

def SBNUCIRFPA_update_nuc(subframe: list | np.ndarray, g: float | np.ndarray, o: float | np.ndarray, eta: float | np.ndarray = 0.01, bias_g: float | np.ndarray = 0, bias_o: float | np.ndarray = 0) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Update the non-uniformity correction coefficients using a subframe.

    This function computes the error between the estimated target and the actual value for a subframe.
    It then updates the gain (g) and offset (o) coefficients using stochastic gradient descent (SGD).

    Args:
        subframe (list | np.ndarray): A subframe extracted from the original frame.
        g (float | np.ndarray): The gain coefficient.
        o (float | np.ndarray): The offset coefficient.
        eta (float | np.ndarray, optional): The learning rate for the update. Defaults to 0.01.
        bias_g (float | np.ndarray, optional): An optional bias term for gain update. Defaults to 0.
        bias_o (float | np.ndarray, optional): An optional bias term for offset update. Defaults to 0.

    Returns:
        tuple[float | np.ndarray, float | np.ndarray]: The updated gain and offset coefficients.
    """
    # Get central pixel of the subframe
    yij = Yij(subframe)
    # Compute target function and error compared to it
    T = kernel_mean_filtering(subframe)
    e = T - Xest(g, yij, o)
    # Update coefficients using SGD
    return sgd_step(g, eta, e * yij, bias_g), sgd_step(o, eta, e, bias_o)

def AdaSBNUCIRFPA(frames: list | np.ndarray, K: float | np.ndarray = 1., k_size=3, offset_only=True) -> np.ndarray:
    """
    Apply the adaptive SBNUC method to a sequence of frames.

    This function iterates through a sequence of frames and applies adaptive non-uniformity correction to each frame
    based on initial coefficients. The corrected frames are collected and returned as a list or numpy array.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        K (float | np.ndarray, optional): The regularization parameter. Defaults to 1.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        list | np.ndarray: A list or array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    coeffs = init_nuc(frames[0])  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames, desc="SBNUCIRFPA processing", unit="frame"):
        frame_est, coeffs = AdaSBNUCIRFPA_frame(frame, coeffs, K, k_size, offset_only)
        all_frame_est.append(frame_est)
    
    print(f"{len(all_frame_est)} frames estimated using AdaSBNUCIRFPA algo")
    return np.array(all_frame_est, frames[0].dtype)


def AdaSBNUCIRFPA_frame(frame: list | np.ndarray, coeffs: dict, K: float | np.ndarray = 1., k_size=3, offset_only = True) -> tuple[list | np.ndarray, dict]:
    """
    Apply the adaptive SBNUC method to a single frame.

    This function processes a single frame using the provided coefficients and updates them adaptively
    based on the local variance of the subframe. It returns the corrected frame along with the updated coefficients.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        coeffs (dict): The coefficients used for the correction.
        K (float | np.ndarray, optional): The regularization parameter. Defaults to 1.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[list | np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    all_Xest = []  # List to store estimated pixel values for the frame
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            subframe = build_kernel(frame, i, j, k_size)  # Extract local kernel around the pixel
            eta = AdaSBNUCIRFPA_eta(K, subframe)  # Calculate adaptive learning rate
            if offset_only:
                # Update coefficients based on local kernel and adaptive learning rate
                _ , coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    subframe, 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta
                )
            else:
                # Update coefficients based on local kernel and adaptive learning rate
                coeffs["g"][i][j], coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    subframe, 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta
                )
            # Estimate corrected pixel value
            all_Xest[i].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
    return np.array(all_Xest, dtype=frame.dtype), coeffs

def AdaSBNUCIRFPA_eta(K: float, subframe: list | np.ndarray, A=1.) -> float:
    """
    Calculate the adaptive learning rate (eta) for SBNUC.

    This function computes the adaptive learning rate based on the variance of the subframe.
    The higher the variance, the smaller the learning rate to avoid over-correction.

    Args:
        K (float): The regularization parameter.
        subframe (list | np.ndarray): The subframe for which the learning rate is being calculated.
        A (float, optional): normalization value. Default is 1. 

    Returns:
        float: The adaptive learning rate.
    """
    var = kernel_var_filtering(subframe)
    return K / (1 + A*(var**2))

def AdaSBNUCIRFPA_reg(frames: list | np.ndarray, K: float | np.ndarray = 1., alpha: float | np.ndarray = 0.4, k_size=3) -> list | np.ndarray:
    """
    Apply the adaptive SBNUC method with regularization to a sequence of frames.

    This function iterates through a sequence of frames and applies adaptive non-uniformity correction with regularization
    to each frame based on initial coefficients. The corrected frames are collected and returned as a list or numpy array.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        K (float | np.ndarray, optional): The regularization parameter. Defaults to 1.
        alpha (float | np.ndarray, optional): The regularization weight. Defaults to 0.4.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.

    Returns:
        list | np.ndarray: A list or array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    coeffs = init_nuc(frames[0])  # Initialize coefficients based on the first frame
    coeffs_n_1 = init_nuc(frames[0])  # Initialize previous frame coefficients
    for frame in frames:
        frame_est, coeffs, coeffs_n_1 = AdaSBNUCIRFPA_reg_frame(frame, coeffs, coeffs_n_1, K, alpha, k_size)
        all_frame_est.append(frame_est)
    return np.array(all_frame_est, dtype=frames[0])

def AdaSBNUCIRFPA_reg_frame(
        frame: list | np.ndarray, 
        coeffs: dict, coeffs_n_1: dict, 
        K: float | np.ndarray = 0.1, alpha: float | np.ndarray = 0.4, 
        k_size=3 , offset_only = True
    ) -> tuple[list | np.ndarray, dict, dict]:
    """
    Apply the adaptive SBNUC method with regularization to a single frame.

    This function processes a single frame using the provided coefficients and updates them adaptively
    with regularization based on the local variance of the subframe and previous frame coefficients.
    It returns the corrected frame along with the updated coefficients.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        coeffs (dict): The coefficients used for the correction.
        coeffs_n_1 (dict): The coefficients from the previous frame for regularization.
        K (float | np.ndarray, optional): The regularization parameter. Defaults to 1.
        alpha (float | np.ndarray, optional): The regularization weight. Defaults to 0.4.
        k_size (int, optional): The size of the kernel (half-width) used for local estimation. Defaults to 3.
        offset_only (bool, optional): Whether to only update the offset coefficient. Defaults to True.

    Returns:
        tuple[list | np.ndarray, dict, dict]: The corrected frame, updated coefficients, and updated previous frame coefficients.
    """
    all_Xest = []  # List to store estimated pixel values for the frame
    for i in range(len(frame)):
        all_Xest.append([])  # Initialize row for estimated values
        for j in range(len(frame[0])):
            subframe = build_kernel(frame, i, j, k_size)  # Extract local kernel around the pixel
            eta = AdaSBNUCIRFPA_eta(K, subframe)  # Calculate adaptive learning rate

            # Swap the values of the previous frame coefficients with the current ones
            g_n_1 = coeffs_n_1["g"][i][j]
            o_n_1 = coeffs_n_1["o"][i][j]
            coeffs_n_1["g"][i][j] = coeffs["g"][i][j]
            coeffs_n_1["o"][i][j] = coeffs["o"][i][j]

            # Update the coefficients using the subframe, learning rate, and regularization term
            if offset_only :
                _ , coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    subframe, 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta,
                    alpha * (coeffs["g"][i][j] - g_n_1),
                    alpha * (coeffs["o"][i][j] - o_n_1)
                )
            else :
                coeffs["g"][i][j], coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    subframe, 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j],
                    eta,
                    alpha * (coeffs["g"][i][j] - g_n_1),
                    alpha * (coeffs["o"][i][j] - o_n_1)
                )

            # Estimate corrected pixel value
            all_Xest[i].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
    return np.array(all_Xest, dtype=frame.dtype), coeffs, coeffs_n_1
