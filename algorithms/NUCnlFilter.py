# NUC algorithm based on scene nonlinear filtering residual estimation
# NUCFnlFilter

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from motion.motion_estimation import *
from algorithms.AdaSBNUCIRFPA import AdaSBNUCIRFPA_eta

def NUCnlFilter(frames, Ts=15, Tg=11, offset_only = True):
    all_frames_est = []
    coeffs = init_nuc(frames[0])
    for i in tqdm(len(frames[1:]), desc="NUCFnlFilter algorithm", unit="frame"):
        if M_n(frame=frames[i], frame_n_1=frames[i-1], Ts=Ts , Tg=Tg):
            frame_est = Xest(g=coeffs['g'], y=frames[i], o=coeffs['o'])
        else:
            frame_est , coeffs['g'], coeffs['o']= NUCnlFilter_frame(frames[i], coeffs, N=i, offset_only=offset_only)
        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames.dtype)

def NUCnlFilter_frame(frame, coeffs, Eij_n_1, K, N:int, Tn, tn, beta=2.42, k_size=3, offset_only=True):
    X_est = Xest(g=coeffs['g'], y=frame, o=coeffs['o'])
    delta = e(frame=frame,
              X_est=X_est,
              N=N,
              Tn=Tn,
              tn=tn,
              Eij=Eij,
              Eij_n_1=Eij_n_1,
              beta=beta
              )
    Eij = delta**2
    for i in range(len(frame)):
         for j in range(len(frame[0])):  
            lr = AdaSBNUCIRFPA_eta(K, build_kernel(frame, i, j, k_size))
            # update nuc
            coeffs['o'][i][j] = sgd_step(coeff=coeffs['o'][i][j], lr=lr, delta=delta)
            if not offset_only:
                coeffs['g'][i][j] = sgd_step(coeff=coeffs['g'][i][j], lr=lr, delta=delta)
    return X_est, coeffs['g'], coeffs['o']


def R_n(N:int, Tn, tn, Eij, Eij_n_1, beta=2.42):
    """
    Temporal residual temperature model estimate

    beta: proportional to temperature change
    Tn : actual temperature of pixel ??? 
    tn : FPA temperature
    """
    dT = 50 # value from engineer : Tmax - Tmin
    gamma = Eij - Eij_n_1
    return (N-2)/N * gamma + (Tn - tn) / dT *beta

def e(frame, X_est, N:int, Tn, tn, Eij, Eij_n_1, beta= 2.42):
    # TODO finish this function integration
    D = frame_gauss_3x3_filtering(frame)
    E =  X_est - D
    # own initiative, not detailed in paper
    # might use binary image as well instead of sobel filter
    phi = np.mean(frame_sobel_3x3_filtering(frame))  # Scene detail magnitude
    W_n = phi / (phi + Tn/tn)
    # W_n = 0.347
    Rn = R_n(N, Tn, tn, Eij, Eij_n_1, beta)
    return exp_window(E, Eij_n_1+Rn, W_n)


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

# TODO finish
def NUCFnlFilter_lr():
    return
