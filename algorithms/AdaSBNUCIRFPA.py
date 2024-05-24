# Adaptive scene-based non-uniformity correction method for infrared focal plane arrays
# AdaSBNUCIRFPA (paper)

from ..common import *
from ..target import *

def SBNUCIRFPA(frames: list | np.ndarray, k_size=3) -> list | np.ndarray:
    """
    Apply the scene-based non-uniformity correction (SBNUC) method to a sequence of frames.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
        k_size (int, optional): The size of the kernel (half-width) used for processing. Defaults to 3.

    Returns:
        list | np.ndarray: A list or array of corrected frames.
    """
    all_frame_est = []
    coeffs = init_nuc(frames[0])
    for frame in frames:
        frame_est, coeffs = SBNUCIRFPA_frame(frame, coeffs, k_size)
        all_frame_est.append(frame_est)
    if type(frames) == np.array:
        return np.array(all_frame_est)
    else:
        return all_frame_est

def SBNUCIRFPA_frame(frame: list | np.ndarray, coeffs: dict, k_size=3) -> tuple[list | np.ndarray, dict]:
    """
    Apply the SBNUC method to a single frame.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
        coeffs (dict): The coefficients used for the correction.
        k_size (int, optional): The size of the kernel (half-width) used for processing. Defaults to 3.

    Returns:
        tuple[list | np.ndarray, dict]: The corrected frame and updated coefficients.
    """
    all_Xest = []
    if type(frame) == np.array:
        pass
    else:
        for i in range(len(frame)):
            all_Xest.append([])
            for j in range(len(frame[0])):
                coeffs["g"][i][j], coeffs["o"][i][j] = SBNUCIRFPA_update_nuc(
                    build_kernel(frame, i, j, k_size), 
                    coeffs["g"][i][j], 
                    coeffs["o"][i][j]
                )
                all_Xest[i].append(Xest(coeffs["g"][i][j], frame[i][j], coeffs["o"][i][j]))
    return all_Xest, coeffs

def AdaSBNUCIRFPA(frames: list | np.ndarray):
    """
    Apply the adaptive SBNUC method to a sequence of frames.

    Args:
        frames (list | np.ndarray): A list or array of frames to be corrected.
    """
    for frame in frames:
        AdaSBNUCIRFPA_frame(frame)
    return

def AdaSBNUCIRFPA_frame(frame: list | np.ndarray):
    """
    Apply the adaptive SBNUC method to a single frame.

    Args:
        frame (list | np.ndarray): A single frame to be corrected.
    """
    return

def SBNUCIRFPA_update_nuc(subframe: list | np.ndarray, g: float | np.ndarray, o: float | np.ndarray, eta: float | np.ndarray = 0.01, bias: float | np.ndarray = 0) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Update the non-uniformity correction coefficients using a subframe.

    Args:
        subframe (list | np.ndarray): A subframe extracted from the original frame.
        g (float | np.ndarray): The gain coefficient.
        o (float | np.ndarray): The offset coefficient.
        eta (float | np.ndarray, optional): The learning rate for the update. Defaults to 0.01.
        bias (float | np.ndarray, optional): An optional bias term. Defaults to 0.

    Returns:
        tuple[float | np.ndarray, float | np.ndarray]: The updated gain and offset coefficients.
    """
    # TODO: Add possibility to change target function?
    # get central pixel of subframe
    yij = Yij(subframe)
    # Compute target function and error compared to it
    T = mean_filtering(subframe)
    e = T - Xest(g, yij, o)
    return sgd_step(g, eta, e * yij, bias), sgd_step(o, eta, e, bias)

def AdaSBNUCIRFPA_lr():
    """
    Placeholder function for adaptive SBNUC learning rate adjustment.
    """
    return

def AdaSBNUCIRFPA_eta():
    """
    Placeholder function for adaptive SBNUC learning rate (eta) adjustment.
    """
    return
