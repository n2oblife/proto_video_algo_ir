# NUC algorithm based on scene nonlinear filtering residual estimation
# NUCFnlFilter

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from algorithms.AdaSBNUCIRFPA import AdaSBNUCIRFPA_eta

def NUCnlFilter_og(frames, Ts=15, Tg=11, offset_only = True):
    """
    Not optimized version of the NUC algorithm based on scene nonlinear filtering residual estimation.

    Args:
        frames (list): List of frames to be processed.
        Ts (int, optional): Pixel change threshold. Defaults to 15.
        Tg (int, optional): Scene movement threshold. Defaults to 11.
        offset_only (bool, optional): Whether to update only the offset coefficient. Defaults to True.

    Returns:
        np.ndarray: Array of estimated frames.
    """
    # Initialize list to store estimated frames
    all_frames_est = []
    # Initialize coefficients for NUC algorithm
    coeffs = init_nuc(frames[0])
    # Loop over frames
    for i in tqdm(range(len(frames[1:])-1), desc="NUCFnlFilter algorithm", unit="frame"):
        # Check for motion using M_n function
        if M_n(frame=frames[i], frame_n_1=frames[i-1], Ts=Ts , Tg=Tg):
            # Apply NUC algorithm to current frame using NUCnlFilter_frame function
            frame_est , coeffs['g'], coeffs['o']= NUCnlFilter_frame(frames[i], coeffs, N=i, offset_only=offset_only)
        else:
            # Estimate current frame using Xest function
            frame_est = Xest(g=coeffs['g'], y=frames[i], o=coeffs['o'])
        # Append estimated frame to list
        all_frames_est.append(frame_est)
    # Convert list to numpy array and return
    return np.array(all_frames_est, dtype=frames.dtype)

def NUCnlFilter_frame(frame, coeffs, Eij_n_1, K, N:int, Tn, tn, beta=2.42, k_size=3, offset_only=True):
    """
    Apply NUC algorithm to a single frame.

    Args:
        frame (np.ndarray): Current frame.
        coeffs (dict): Coefficients for NUC algorithm.
        Eij_n_1 (np.ndarray): Previous residual error.
        K (float): Learning rate parameter.
        N (int): Current frame index.
        Tn (float): Actual temperature of pixel.
        tn (float): FPA temperature.
        beta (float, optional): Proportional to temperature change. Defaults to 2.42.
        k_size (int, optional): Kernel size for filtering. Defaults to 3.
        offset_only (bool, optional): Whether to update only the offset coefficient. Defaults to True.

    Returns:
        tuple: Estimated frame, updated gain coefficient, updated offset coefficient, and residual error.
    """
    # Estimate current frame using Xest function
    X_est = Xest(g=coeffs['g'], y=frame, o=coeffs['o'])
    # Calculate residual error using e function
    delta, Eij_n = e(frame=frame, X_est=X_est, N=N, Tn=Tn, tn=tn, Eij_n_1=Eij_n_1, beta=beta)
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            # Calculate learning rate using AdaSBNUCIRFPA_eta function
            lr = AdaSBNUCIRFPA_eta(K, build_kernel(frame, i, j, k_size))
            # Update NUC coefficients using sgd_step function
            coeffs['o'][i][j] = sgd_step(coeff=coeffs['o'][i][j], lr=lr, delta=2*delta)
            if not offset_only:
                coeffs['g'][i][j] = sgd_step(coeff=coeffs['g'][i][j], lr=lr, delta=2*delta*frame[i][j])
    # Calculate residual error
    if N>=2:
        Eij = Eij_n**2
    else:
        Eij = delta**2
    # Return estimated frame, updated coefficients, and residual error
    return X_est, coeffs['g'], coeffs['o'], Eij

def R_n(N:int, Eij, Eij_n_1, Tn=0, tn=0, beta=2.42):
    """
    Calculate temporal residual temperature model estimate.

    Args:
        N (int): Current frame index.
        Eij (np.ndarray): Current residual error.
        Eij_n_1 (np.ndarray): Previous residual error.
        Tn (float, optional): Actual temperature of pixel. Defaults to 0.
        tn (float, optional): FPA temperature. Defaults to 0.
        beta (float, optional): Proportional to temperature change. Defaults to 2.42.

    Returns:
        float: Temporal residual temperature model estimate.
    """
    # Calculate temporal residual temperature model estimate using given formula
    dT = 50 # value from engineers : Tmax - Tmin in cam
    gamma = Eij - Eij_n_1
    return (N-2)/N * gamma + (Tn - tn) / dT *beta

def e(frame, X_est, Eij_n_1, N:int, Tn=0, tn=0, beta= 2.42):
    """
    Calculate residual error.

    Args:
        frame (np.ndarray): Current frame.
        X_est (np.ndarray): Estimated frame.
        Eij_n_1 (np.ndarray): Previous residual error.
        N (int): Current frame index.
        Tn (float, optional): Actual temperature of pixel. Defaults to 0.
        tn (float, optional): FPA temperature. Defaults to 0.
        beta (float, optional): Proportional to temperature change. Defaults to 2.42.

    Returns:
        tuple: Residual error and updated residual error.
    """
    # Apply Gaussian filter to current frame
    D = frame_gauss_3x3_filtering(frame)
    # Calculate residual error
    E =  X_est - D
    # Calculate scene detail magnitude using Sobel filter
    phi = np.mean(frame_sobel_3x3_filtering(frame))
    # Calculate weight parameter
    if Tn==0 and tn==0:
        W_n = 0.347 #value given in paper
    else:
        W_n = phi / (phi + Tn/tn)
    # Calculate temporal residual temperature model estimate using R_n function
    Rn = R_n(N, Tn, tn, E, Eij_n_1, beta)
    # Update residual error using exponential window filtering
    return exp_window(E, Eij_n_1+Rn, W_n), E

def M_n(frame, frame_n_1, Ts=15, Tg=11)->bool:
    """
    Check for motion using difference image and binary image.

    Args:
        frame (np.ndarray): Current frame.
        frame_n_1 (np.ndarray): Previous frame.
        Ts (int, optional): Pixel change threshold. Defaults to 15.
        Tg (int, optional): Scene movement threshold. Defaults to 11.

    Returns:
        bool: True if motion is detected, False otherwise.
    """
    # Calculate difference image
    df = np.abs(frame - frame_n_1)
    # Calculate binary image
    Bs = np.where(df >= Ts, 1, 0)
    # Calculate global motion parameter
    Mo = np.sum(Bs)
    # Check for motion
    return Mo >= Tg

def NUCnlFilter(frames, Ts=15, Tg=11, offset_only = True):
    """
    Optimized version of the NUC algorithm based on scene nonlinear filtering residual estimation.

    Args:
        frames (list): List of frames to be processed.
        Ts (int, optional): Pixel change threshold. Defaults to 15.
        Tg (int, optional): Scene movement threshold. Defaults to 11.
        offset_only (bool, optional): Whether to update only the offset coefficient. Defaults to True.

    Returns:
        np.ndarray: Array of estimated frames.
    """
    # Initialize list to store estimated frames
    all_frames_est = []
    # Initialize coefficients for NUC algorithm
    coeffs = init_nuc(frames[0])
    # Initialize previous residual error
    Eij_n = np.zeros(frames[0].shape, dtype=frames[0].dtype)
    # Loop over frames
    for i in tqdm(range(1, len(frames[1:])-1), desc="NUCFnlFilter algorithm", unit="frame"):
        # Check for motion using M_n function
        if M_n(frame=frames[i], frame_n_1=frames[i-1], Ts=Ts , Tg=Tg):
            # Apply NUC algorithm to current frame using NUCnlFilter_frame_array function
            frame_est , coeffs['g'], coeffs['o'], Eij_n= NUCnlFilter_frame_array(
                frame=frames[i],
                coeffs=coeffs,
                Eij_n_1=Eij_n,
                N=i,
                offset_only=offset_only
            )
        else:
            # Estimate current frame using Xest function
            frame_est = Xest(g=coeffs['g'], y=frames[i], o=coeffs['o'])
        # Append estimated frame to list
        all_frames_est.append(frame_est)
    # Convert list to numpy array and return
    return np.array(all_frames_est, dtype=frames.dtype)

def NUCnlFilter_frame_array(frame, coeffs, Eij_n_1, N:int,Tn=0, tn=0, beta=2.42, k_size=3, offset_only=True):
    """
    Apply NUC algorithm to a single frame using numpy array operations.

    Args:
        frame (np.ndarray): Current frame.
        coeffs (dict): Coefficients for NUC algorithm.
        Eij_n_1 (np.ndarray): Previous residual error.
        N (int): Current frame index.
        Tn (float, optional): Actual temperature of pixel. Defaults to 0.
        tn (float, optional): FPA temperature. Defaults to 0.
        beta (float, optional): Proportional to temperature change. Defaults to 2.42.
        k_size (int, optional): Kernel size for filtering. Defaults to 3.
        offset_only (bool, optional): Whether to update only the offset coefficient. Defaults to True.

    Returns:
        tuple: Estimated frame, updated gain coefficient, updated offset coefficient, and residual error.
    """
    # Estimate current frame using Xest function
    X_est = Xest(g=coeffs['g'], y=frame, o=coeffs['o'])
    # Calculate residual error using e function
    delta, Eij_n = e(frame=frame, X_est=X_est, N=N, Tn=Tn, tn=tn, Eij_n_1=Eij_n_1, beta=beta)
    # Calculate learning rate using NUCFnlFilter_lr function
    lr = NUCFnlFilter_lr(frame=frame, k_size=k_size)
    # Update NUC coefficients using sgd_step function
    coeffs['o'] = sgd_step(coeff=coeffs['o'], lr=lr, delta=2*delta)
    if not offset_only:
        coeffs['g'] = sgd_step(coeff=coeffs['g'], lr=lr, delta=2*delta*frame)
    # Calculate residual error
    if N>=2:
        Eij = Eij_n**2
    else:
        Eij = delta**2
    # Return estimated frame, updated coefficients, and residual error
    return X_est, coeffs['g'], coeffs['o'], Eij

def NUCFnlFilter_lr(
        frame: list | np.ndarray,
        K: float | np.ndarray = 0.001,
        k_size=3,
        A=1
    ):
    """
    Calculate adaptive learning rate for NUC algorithm.

    Adapted from article : A Novel Infrared Focal Plane Non-Uniformity Correction Method Based on Co-Occurrence Filter and Adaptive Learning Rate
    L. Li, Q. Li, H. Feng, Z. Xu, and Y. Chen, “A novel infrared focal plane non-uniformity correction method based on cooccurrence filter and adaptive learning rate,” IEEE Access 7, 40941–40950 (2019)

    Args:
        frame (list | np.ndarray): Current frame.
        K (float | np.ndarray, optional): Learning rate parameter. Defaults to 0.001.
        k_size (int, optional): Kernel size for filtering. Defaults to 3.
        A (int, optional): Parameter for learning rate calculation. Defaults to 1.

    Returns:
        float | np.ndarray: Learning rate.
    """
    # Calculate learning rate using given formula
    return K / (1+A*frame_var_filtering(frame, k_size)**2)
