# NUC algorithm based on scene nonlinear filtering residual estimation
# NUCFnlFilter

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *
from algorithms.AdaSBNUCIRFPA import AdaSBNUCIRFPA_eta

def NUCnlFilter_og(frames, Ts=15, Tg=11, offset_only = True):
    """
    Not optimized

    Args:
        frames (_type_): _description_
        Ts (int, optional): _description_. Defaults to 15.
        Tg (int, optional): _description_. Defaults to 11.
        offset_only (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    all_frames_est = []
    coeffs = init_nuc(frames[0])
    for i in tqdm(range(len(frames[1:])-1), desc="NUCFnlFilter algorithm", unit="frame"):
        if M_n(frame=frames[i], frame_n_1=frames[i-1], Ts=Ts , Tg=Tg):
            frame_est , coeffs['g'], coeffs['o']= NUCnlFilter_frame(frames[i], coeffs, N=i, offset_only=offset_only)
        else:
            frame_est = Xest(g=coeffs['g'], y=frames[i], o=coeffs['o'])
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames.dtype)

def NUCnlFilter_frame(frame, coeffs, Eij_n_1, K, N:int, Tn, tn, beta=2.42, k_size=3, offset_only=True):
    X_est = Xest(g=coeffs['g'], y=frame, o=coeffs['o'])
    delta, Eij_n = e(frame=frame,
              X_est=X_est,
              N=N,
              Tn=Tn,
              tn=tn,
              Eij_n_1=Eij_n_1,
              beta=beta
              )
    if N>=2:
        Eij = Eij_n**2
    else:
        Eij = delta**2
    for i in range(len(frame)):
         for j in range(len(frame[0])):  
            lr = AdaSBNUCIRFPA_eta(K, build_kernel(frame, i, j, k_size))
            # update nuc
            coeffs['o'][i][j] = sgd_step(coeff=coeffs['o'][i][j], lr=lr, delta=2*delta)
            if not offset_only:
                coeffs['g'][i][j] = sgd_step(coeff=coeffs['g'][i][j], lr=lr, delta=2*delta*frame[i][j])
    return X_est, coeffs['g'], coeffs['o'], Eij


def R_n(N:int, Eij, Eij_n_1, Tn=0, tn=0, beta=2.42):
    """
    Temporal residual temperature model estimate

    beta: proportional to temperature change
    Tn : actual temperature of pixel ???
    tn : FPA temperature
    """
    dT = 50 # value from engineers : Tmax - Tmin in cam
    gamma = Eij - Eij_n_1
    return (N-2)/N * gamma + (Tn - tn) / dT *beta

def e(frame, X_est, Eij_n_1, N:int, Tn=0, tn=0, beta= 2.42):
    # TODO finish this function integration
    D = frame_gauss_3x3_filtering(frame)
    E =  X_est - D
    # own initiative, not detailed in paper
    # might use binary image as well instead of sobel filter
    phi = np.mean(frame_sobel_3x3_filtering(frame))  # Scene detail magnitude
    if Tn==0 and tn==0:
        W_n = 0.347 #value given in paper
    else:
        W_n = phi / (phi + Tn/tn)
    Rn = R_n(N, Tn, tn, E, Eij_n_1, beta)
    return exp_window(E, Eij_n_1+Rn, W_n), E


def M_n(frame, frame_n_1, Ts=15, Tg=11)->bool:
    """
    Motion judgement parameter

    Args:
        frame (_type_): _description_
        frame_n_1 (_type_): _description_
        Ts (_type_): change threshold
        Tg (_type_): scene movement threshold
    """
    df = np.abs(frame - frame_n_1) # difference image
    Bs = np.where(df >= Ts, 1, 0) # binary image
    Mo = np.sum(Bs) # global motion parameter
    return Mo >= Tg



def NUCnlFilter(frames, Ts=15, Tg=11, offset_only = True):
    all_frames_est = []
    coeffs = init_nuc(frames[0])
    Eij_n = np.zeros(frames[0].shape, dtype=frames[0].dtype)
    for i in tqdm(range(1, len(frames[1:])-1), desc="NUCFnlFilter algorithm", unit="frame"):
        if M_n(frame=frames[i], frame_n_1=frames[i-1], Ts=Ts , Tg=Tg):
            
            frame_est , coeffs['g'], coeffs['o'], Eij_n= NUCnlFilter_frame_array(
                frame=frames[i], 
                coeffs=coeffs, 
                Eij_n_1=Eij_n,
                N=i, 
                offset_only=offset_only
                )
            
        else:
            frame_est = Xest(g=coeffs['g'], y=frames[i], o=coeffs['o'])
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames.dtype)

def NUCnlFilter_frame_array(frame, coeffs, Eij_n_1, N:int,Tn=0, tn=0, beta=2.42, k_size=3, offset_only=True):
    X_est = Xest(g=coeffs['g'], y=frame, o=coeffs['o'])
    delta, Eij_n = e(frame=frame,
              X_est=X_est,
              N=N,
              Tn=Tn,
              tn=tn,
              Eij_n_1=Eij_n_1,
              beta=beta
              )
    if N>=2:
        Eij = Eij_n**2
    else:
        Eij = delta**2
    lr = NUCFnlFilter_lr(frame=frame, k_size=k_size)
    # update nuc
    coeffs['o'] = sgd_step(coeff=coeffs['o'], lr=lr, delta=2*delta)
    if not offset_only:
        coeffs['g'] = sgd_step(coeff=coeffs['g'], lr=lr, delta=2*delta*frame)
    return X_est, coeffs['g'], coeffs['o'], Eij

def NUCFnlFilter_lr(
        frame: list | np.ndarray, 
        K: float | np.ndarray = 0.001, 
        k_size=3, 
        A=1
    ):
    """
    Adaptive learning rate similar to the one in AdaSBNUCIRFPA with a condition in case of motion or not

    Adapted from article : A Novel Infrared Focal Plane Non-Uniformity Correction Method Based on Co-Occurrence Filter and Adaptive Learning Rate
    L. Li, Q. Li, H. Feng, Z. Xu, and Y. Chen, “A novel infrared focal plane non-uniformity correction method based on cooccurrence filter and adaptive learning rate,” IEEE Access 7, 40941–40950 (2019)
    """
    return K / (1+A*frame_var_filtering(frame, k_size)**2)
