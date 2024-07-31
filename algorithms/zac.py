import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
import algorithms as alg

def zac_NUCnlFilter(frames: list | np.ndarray):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)     # Initialize coefficients based on the first frame
    coeffs = init_nuc(frames[0])
    frame_est_n_1 = frames[0]
    Eij_n = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    for i in tqdm(range(1,len(frames[1:])-1), desc="zac NUCnlFilter algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frames[i], img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            frame_est , coeffs['g'], coeffs['o'], Eij_n= alg.NUCnlFilter.NUCnlFilter_frame_array(
                frame=frame_est, 
                coeffs=coeffs, 
                Eij_n_1=Eij_n,
                N=i
                )        
        else:
            frame_est = Xest(g=coeffs['g'], y=frame_est, o=coeffs['o'])
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)

def zac_smartCam(frames: list | np.ndarray, alpha=0.01, alpha_k=2**(-8)):
    all_frames_est = []      # List to store estimated (corrected) frames
    m_k = np.zeros(frames[0].shape, dtype=frames.dtype)
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    low_passed = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)

    for frame in tqdm(frames, desc="zac smartCam algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc, alpha=alpha)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            frame_est, m_k, low_passed = alg.SBNUC_smartCam.SBNUC_smartCam_pipeA_frame(
                frame=frame_est, m_k=m_k, alpha=alpha_k, low_passed=low_passed)  
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)

def zac_AdaSBNUCIRFPA_window(
        frames: list | np.ndarray, alpha=0.01, 
        K: float | np.ndarray = 0.01,
        offset_only=True
    ):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)
    smoothed = np.zeros(frames[0].shape, dtype=frames.dtype)
    coeffs = init_nuc(frames[0])

    for frame in tqdm(frames, desc="zac AdaSBNUCIRFPA windowed algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # adaptative learning rate
            eta = K / (1+frame_var_filtering(frame_est)**2)
            eta_valid = (0<= eta.all() <=1) if isinstance(eta, np.ndarray) else (0<= eta <=1)
            if eta_valid:
                smoothed = frame_exp_window_filtering(image=frame_est, low_passed=smoothed, alpha=eta)
            else:
                smoothed = frame_exp_window_filtering(image=frame_est, low_passed=smoothed, alpha=alpha)
            frame_est, coeffs = alg.AdaSBNUCIRFPA.SBNUCIRFPA_frame_array(frame=frame_est,
                                                    estimation_frame=smoothed,
                                                    eta=eta,
                                                    coeffs=coeffs,
                                                    offset_only=offset_only)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)

def zac_AdaSBNUCIRFPA_reg(
        frames: list | np.ndarray, alpha=0.01, 
        K: float | np.ndarray = 0.01,
        offset_only=True
    ):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)
    smoothed = np.zeros(frames[0].shape, dtype=frames.dtype)
    coeffs, coeffs_n_1 = init_nuc(frames[0]), init_nuc(frames[0])

    for frame in tqdm(frames, desc="zac AdaSBNUCIRFPA windowed algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # adaptative learning rate
            eta = alg.AdaSBNUCIRFPA.AdaSBNUCIRFPA_eta(frame=frame_est, K=K)
            # smoothed = frame_exp_window_filtering(image=img_nuc, low_passed=smoothed, alpha=alpha)
            smoothed = frame_mean_filtering(image=frame_est)
            frame_est, coeffs, coeffs_n_1 = alg.AdaSBNUCIRFPA.AdaSBNUCIRFPA_reg_frame_array(frame=frame_est,
                                                    estimation_frame=smoothed,
                                                    coeffs=coeffs,
                                                    coeffs_n_1=coeffs_n_1,
                                                    eta=eta,
                                                    alpha=alpha,
                                                    offset_only=offset_only)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)

def zac_AdaSBNUCIRFPA_mean(
        frames: list | np.ndarray, 
        K: float | np.ndarray = 0.1, k_size=3,
        offset_only=True
    ):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)

    for frame in tqdm(frames, desc="zac AdaSBNUCIRFPA meaned algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # adaptative learning rate
            eta = K / (1+frame_var_filtering(frame)**2)
            smoothed = frame_mean_filtering(image=frame_est, k_size=k_size)
            frame_est, coeffs = alg.AdaSBNUCIRFPA.SBNUCIRFPA_frame_array(frame=frame_est,
                                                    estimation_frame=smoothed,
                                                    eta=eta,
                                                    coeffs=coeffs,
                                                    bias_o=2**13,
                                                    offset_only=offset_only)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_RobustNUCIRFPA(frames: list | np.ndarray, alpha_m=0.05, offset_only=True):
    all_frames_est = []      # List to store estimated (corrected) frames
    coeffs = init_nuc(frames[0])
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)

    for frame in tqdm(frames, desc="zac Adaptive RobustNUCIRFPA algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # Estimate the current frame using the previous frame
            if offset_only:
                frame_est, _, coeffs['o'] = alg.RobustNUCIRFPA.AdaRobustNUCIRFPA_frame(
                    frame=frame_est,
                    coeffs=coeffs,
                    frame_n_1=frame_est_n_1,
                    alpha_m=alpha_m
                )
            else:
                frame_est, coeffs['g'], coeffs['o'] = alg.RobustNUCIRFPA.AdaRobustNUCIRFPA_frame(
                    frame=frame_est,
                    coeffs=coeffs,
                    frame_n_1=frame_est_n_1,
                    alpha_m=alpha_m
                )  
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_SBNUCcomplement(frames: list | np.ndarray):
    all_frames_est = []      # List to store estimated (corrected) frames
    Cn = frames[0]  # Cumulative mean image
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)

    for frame in tqdm(frames[1:], desc="zac SBNUCcomplement algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            frame_est, Cn = alg.SBNUCcomplement.SBNUCcomplement_frame(
                frame=frame_est, frame_n_1=frame_est_n_1, Cn=Cn)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)



def zac_CstStatSBNUC(frames: list | np.ndarray, threshold=0., alpha=0.01, offset_only=True):
    # TODO check overflow
    all_frames_est = []      # List to store estimated (corrected) frames
    frame_est_n_1 = frames[0]  # Cumulative mean image
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    corr_coeffs = init_nuc(frames[0])
    sbnuc_coeffs = alg.SBNUCrgGLMS.init_S_M(frames[0])


    for frame in tqdm(frames[1:], desc="zac CstStatSBNUC algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # Estimate the current frame using the previous frame
            frame_est, corr_coeffs, sbnuc_coeffs = alg.SBNUCrgGLMS.CstStatSBNUC_frame_array(
                frame=frame_est, frame_n_1=frame_est_n_1, 
                corr_coeffs=corr_coeffs, sbnuc_coeffs=sbnuc_coeffs, 
                threshold=threshold, alpha=alpha, offset_only=offset_only
            )
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_SBNUCLMS(frames: list | np.ndarray, K=0.1, M=0.5, threshold=0., k_size=3, offset_only=True):
    all_frames_est = []      # List to store estimated (corrected) frames
    frame_est_n_1 = frames[0]  # Cumulative mean image
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    corr_coeffs = init_nuc(frames[0])  # Initialize the correction coefficients based on the first frame
    Z = np.full(frames[0].shape, np.inf, dtype=frames[0].dtype)  # Initialize the Z matrix with infinite values


    for frame in tqdm(frames[1:], desc="zac SBNUCLMS algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            # Estimate the current frame using SBNUCLMS_frame function
            frame_est, corr_coeffs = alg.SBNUCrgGLMS.SBNUCLMS_frame_array(
                frame_est, corr_coeffs, Z, K, M, threshold=threshold, k_size=k_size, offset_only=offset_only)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_AdaSBNUCif_reg(frames: list | np.ndarray, lr=0.05, algo='FourierShift', offset_only=True):
    all_frames_est = []      # List to store estimated (corrected) frames
    frame_est_n_1 = frames[0]  # Cumulative mean image
    img_nuc = np.zeros(frames[0].shape, dtype=frames.dtype)
    coeffs = init_nuc(frames[0])  # Initialize the correction coefficients based on the first frame

    for frame in tqdm(frames[1:], desc="zac AdaSBNUCif_reg algorithm", unit="frame"):
        frame_est, img_nuc = alg.morgan.morgan_frame(frame, img_nuc)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            frame_est, coeffs = alg.SBNUCif_reg.AdaSBNUCif_reg_frame_array(
                frame_est, frame_est_n_1, coeffs, lr, algo, offset_only)
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_morgan(frames: list | np.ndarray, alpha=0.01, alpha_k=0.001, moving_rate=0.1):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc_og = np.zeros(frames[0].shape, dtype=frames.dtype)
    img_nuc_2 = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_thr = frames[0].shape[0] * frames[0].shape[1] * moving_rate

    for frame in tqdm(frames, desc="zac morgan algorithm", unit="frame"):
        frame_est, img_nuc_og = alg.morgan.morgan_frame(frame=frame, img_nuc=img_nuc_og, alpha=alpha)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1, Tg=frame_thr):
            frame_est, img_nuc_2= alg.morgan.morgan_frame(
                frame=frame_est, img_nuc=img_nuc_2, alpha=alpha_k) 
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)


def zac_Adamorgan(frames: list | np.ndarray, alpha=2**-3, K=0.01, A=0.2, moving_rate=0.1):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc_og = np.zeros(frames[0].shape, dtype=frames.dtype)
    img_nuc_2 = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_est_n_1 = np.zeros(frames[0].shape, dtype=frames.dtype)
    frame_thr = frames[0].shape[0] * frames[0].shape[1] * moving_rate

    for frame in tqdm(frames, desc="zac morgan algorithm", unit="frame"):
        frame_est, img_nuc_og = alg.morgan.morgan_frame(frame=frame, img_nuc=img_nuc_og, alpha=alpha)
        if alg.NUCnlFilter.M_n(frame=frame_est, frame_n_1=frame_est_n_1, Tg=frame_thr):
            frame_est, img_nuc_2= alg.morgan.Adamorgan_frame(
                frame=frame_est, img_nuc=img_nuc_2, K=K, A=A) 
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)