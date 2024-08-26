import numpy as np
from tqdm import tqdm
from motion.motion_estimation import *
from utils.target import *
from utils.data_handling import save_frames
from algorithms.NUCnlFilter import M_n


#TOOD debug
import matplotlib.pyplot as plt

def morgan(frames: list | np.ndarray, alpha=0.001):
    """
    Apply the Morgan algorithm to a sequence of frames for non-uniformity correction (NUC).

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.001.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan algo processing", unit="frame"):
        frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_frame(frame, img_nuc=0, alpha=0.01):
    """
    Perform SBNUCIRFPA on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    frame_est = frame + 2**13 - img_nuc

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est).astype(frame.dtype), img_nuc.astype(frame.dtype)

def morgan_moving(frames: list | np.ndarray, alpha=0.01, moving_rate=0.2):
    """
    Apply the Morgan algorithm with motion estimation to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.
        moving_rate (float, optional): Rate of the image change to consider movement. Defaults to 0.2.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    frame_thr = frames[0].shape[0] * frames[0].shape[1] * moving_rate
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame
    frame_est_n_1, img_nuc = morgan_frame(frames[0], img_nuc, alpha)
    img_nuc_n_1 = img_nuc 

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan moving algo processing", unit="frame"):
        frame_est, img_nuc = morgan_frame(frame, img_nuc, alpha)
        # Perform motion estimation to decide if the frame should be corrected
        if M_n(frame=frame_est, frame_n_1=frame_est_n_1, Tg=frame_thr):  # motion estimation might be better with estimation frame comparison
            # update variables
            img_nuc_n_1 = img_nuc
            frame_est_n_1 = frame_est
            # add estimated frame to the rest
            all_frame_est.append(frame_est)

        else:
            # update variables
            img_nuc = img_nuc_n_1
            frame_est_n_1 = frame - img_nuc_n_1
            # add estimated frame to the rest
            all_frame_est.append(frame_est_n_1)
        
    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt(frames: list | np.ndarray, alpha=0.01):
    """
    Apply the Morgan algorithm with additional filtering to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_frame(frame, img_nuc, alpha=0.01):
    """
    Perform SBNUCIRFPA with additional filtering on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame with additional filtering
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    img_nuc_bas = frame_gauss_3x3_filtering(img_nuc)
    img_nuc_haut = frame_laplacian_3x3_filtering(img_nuc)
    frame_est = frame + 2**13 - (img_nuc_bas + img_nuc_haut)

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def morgan_filt_haut(frames: list | np.ndarray, alpha=0.01):
    """
    Apply the Morgan algorithm with high-pass filtering to a sequence of frames.

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        alpha (float, optional): Learning rate for the correction. Defaults to 0.01.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = frames[0] - frame_mean_filtering(frames[0], 5)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan filtering haut algo processing", unit="frame"):
        frame_est, img_nuc = morgan_filt_haut_frame(frame, img_nuc, alpha)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def morgan_filt_haut_frame(frame, img_nuc, alpha=0.01):
    """
    Perform SBNUCIRFPA with high-pass filtering on a single frame.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        alpha (float): The learning rate for the correction.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Perform Morgan algorithm on the frame with high-pass filtering
    img_nuc = alpha * frame + (1 - alpha) * img_nuc
    img_nuc_haut = img_nuc - frame_mean_filtering(img_nuc, 5)
    frame_est = frame + 2**13 - img_nuc_haut

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est), img_nuc

def Adamorgan(frames: list | np.ndarray, K=2**-3, A=2**-8):
    """
    Apply the Adamorgan algorithm to a sequence of frames for non-uniformity correction (NUC).

    Args:
        frames (list | np.ndarray): List or array of frames to be corrected.
        K (float, optional): Learning rate parameter. Defaults to 0.01.
        A (float, optional): Parameter for adaptive learning rate. Defaults to 0.2.

    Returns:
        np.ndarray: Array of corrected frames.
    """
    all_frame_est = []  # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)  # Initialize coefficients based on the first frame

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="Adamorgan algo processing", unit="frame"):
        frame_est, img_nuc = Adamorgan_frame(frame, img_nuc, K=K, A=A)
        all_frame_est.append(frame_est)

    return np.array(all_frame_est, dtype=frames[0].dtype)

def Adamorgan_frame(frame, img_nuc=0, K=2**-3, A=2**-8):
    """
    Perform SBNUCIRFPA on a single frame using the Adamorgan algorithm.

    Args:
        frame (np.ndarray): The frame to perform SBNUCIRFPA on.
        img_nuc (np.ndarray): The coefficients to use for the correction.
        K (float): Learning rate parameter.
        A (float): Parameter for adaptive learning rate.

    Returns:
        np.ndarray: The corrected frame.
    """
    # Calculate the adaptive learning rate
    eta = K / (1 + A * frame_var_filtering(frame)**2)
    # breakpoint()

    # Perform Morgan algorithm on the frame with adaptive learning rate
    img_nuc = eta * frame + (1 - eta) * img_nuc
    frame_est = frame + 2**13 - img_nuc

    # Use numpy's where function to set values less than 0 to 0
    return np.where(frame_est < 0, 0, frame_est).astype(frame.dtype), img_nuc.astype(frame.dtype)


# TODO maybe lower alpha a bit because overlaping zone too effective
def morgan_overlap(frames: list | np.ndarray, alpha=0.025, border_min=16, border_max=64, algo='FourierShift',):
    all_frames_est = []
    frame_n_1 = frames[0]  # Initialize with the first frame
    img_nuc = np.full(shape=frames[0].shape, dtype=frames[0].dtype, fill_value=2**13)
    
    #TODO debug
    overlap_list = []
    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="morgan overlap algo processing", unit="frame"):
        frame_est, img_nuc, overlap_nuc = morgan_overlap_frame(
            frame=frame, frame_n_1=frame_n_1,
            img_nuc=img_nuc, alpha=alpha, algo=algo, border_min=border_min, border_max=border_max
        )
        all_frames_est.append(frame_est)
        frame_n_1 = frame  # Update the previous frame for motion detection

        #TODO debug
        overlap_value = 1 if overlap_nuc else 0
        overlap_list.append(overlap_value)

    plt.plot(overlap_list)
    plt.show()

    return np.array(all_frames_est, dtype=frames[0].dtype)

def morgan_overlap_frame(
        frame: list | np.ndarray, frame_n_1: list | np.ndarray,
        img_nuc=0, alpha=0.01, 
        algo='FourierShift', border_min=16, border_max=64
    ):

    #estimate the frame anyway with not yet updated img_nuc
    temp_frame = frame + 2**13 - img_nuc

    # Estimate the motion vector between the previous frame and the current frame both enhanced
    if algo == "OptFlow":
        di, dj = motion_estimation_frame(prev_frame=frame_n_1.astype(np.uint8), curr_frame=temp_frame.astype(np.uint8), algo=algo).astype(np.int16)
    else :
        di, dj = motion_estimation_frame(prev_frame=frame_n_1, curr_frame=temp_frame, algo=algo)

    # Update threshold from the paper
    update_nuc = ((border_min <= np.sqrt(di**2 + dj**2) <= border_max) and dj!=0)

    # Update the coefficients and estimate the corrected pixel values
    if update_nuc:        
        # define overlaping areas
        idi_min_n_1, idi_max_n_1 = max(0,-di), min(frame.shape[0]-1, frame.shape[0]-1 - di)
        jdj_min_n_1, jdj_max_n_1 = max(0,-dj), min(frame.shape[1]-1, frame.shape[1]-1 - dj)    
        
        idi_min, idi_max = max(0,di), min(frame.shape[0]-1, frame.shape[0]-1 + di)
        jdj_min, jdj_max = max(0,dj), min(frame.shape[1]-1, frame.shape[1]-1 + dj)

        # # #get error from one way and then the other way
        # Eij = (2**13 + frame[idi_min:idi_max, jdj_min:jdj_max] - frame_n_1[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1]).astype(frame.dtype)
        # Eij_p = (2**13 + frame_n_1[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1] - frame[idi_min:idi_max, jdj_min:jdj_max]).astype(frame.dtype)

        # # Create a mask of the same shape as frame that is True where Eij is not computed and False otherwise
        # mask = np.zeroes(frame.shape, dtype=bool)
        # mask[idi_min:idi_max, jdj_min:jdj_max] = False

        # # Perform Morgan algorithm on the frame
        # img_nuc[idi_min:idi_max, jdj_min:jdj_max] = alpha * Eij + (1 - alpha) * img_nuc[idi_min:idi_max, jdj_min:jdj_max]
        # img_nuc[mask] = alpha * Eij_p + (1 - alpha) * img_nuc[mask]
        # img_nuc[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1] = alpha * Eijp + (1 - alpha) * img_nuc[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1]

        # define overlaping area of both error matrix
        idi_full_min, idi_full_max = max(idi_min_n_1, idi_min), min(idi_max_n_1, idi_max)
        jdi_full_min, jdi_full_max = max(jdj_min_n_1, jdj_min), min(jdj_max_n_1, jdj_max)
        
        #get error from one way and then the other way
        # EijFull = np.full(shape=frame.shape, dtype=frame.dtype, fill_value=-1) #pb of overflow
        EijFull = np.full(shape=frame.shape, dtype=np.int32, fill_value=-1)
        EijFull[idi_min:idi_max, jdj_min:jdj_max] = (2**13 + frame[idi_min:idi_max, jdj_min:jdj_max] - frame_n_1[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1]).astype(frame.dtype)
        EijFull[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1] += (2**13 + frame_n_1[idi_min_n_1:idi_max_n_1, jdj_min_n_1:jdj_max_n_1] - frame[idi_min:idi_max, jdj_min:jdj_max]).astype(frame.dtype)
        EijFull[idi_full_min:idi_full_max, jdi_full_min:jdi_full_max] = (EijFull[idi_full_min:idi_full_max, jdi_full_min:jdi_full_max] / 2).astype(frame.dtype)

        # Perform gaussian window on the frame
        img_nuc = np.where(
            EijFull != -1, 
            alpha*EijFull + (1-alpha)*img_nuc,
            img_nuc
            )

    #estimate the frame anyway with updated img_nuc or not
    # frame_est = frame + 2**13 - img_nuc

    #TODO debug returun bool
    return np.where(temp_frame < 0, 0, temp_frame).astype(frame.dtype), img_nuc.astype(frame.dtype), update_nuc