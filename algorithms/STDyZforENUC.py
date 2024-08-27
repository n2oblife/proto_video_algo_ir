import numpy as np
from tqdm import tqdm
from motion.motion_estimation import *


def STDyZforENUC(
        frames: list | np.ndarray, 
        alpha_gauss=0.0004275, 
        alpha_overlap=0.035, 
        border_gauss = 2, border_min=16, border_max=128, 
        algo='FourierShift'
    ):
    all_frames_est = []
    frame_temp_n_1, frame_est_n_1, og_frame_n_1 = frames[0], frames[0], frames[0]  # Initialize with the first frame
    img_nuc_gauss = np.full(shape=frames[0].shape, dtype=frames[0].dtype, fill_value=2**13)
    img_nuc_overlap = np.full(shape=frames[0].shape, dtype=frames[0].dtype, fill_value=2**13)

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[1:], desc="STDyZforENUC algo processing", unit="frame"):
        
        frame_est, frame_temp_n_1, img_nuc_gauss, img_nuc_overlap = STDyZforENUC_frame(
            frame=frame, og_frame_n_1=og_frame_n_1,
            frame_temp_n_1=frame_temp_n_1, enhanced_frame_n_1=frame_est_n_1, 
            alpha_gauss=alpha_gauss, alpha_overlap=alpha_overlap,
            img_nuc_gauss=img_nuc_gauss, img_nuc_overlap=img_nuc_overlap,
            border_gauss=border_gauss, border_min=border_min, border_max=border_max,
            algo=algo,
        )

        # Update the previous frame for motion detection and error computation
        frame_est_n_1 = frame_est
        og_frame_n_1 = frame

        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)


def STDyZforENUC_frame(
        frame: list | np.ndarray, og_frame_n_1: list | np.ndarray,
        frame_temp_n_1: list | np.ndarray, enhanced_frame_n_1: list | np.ndarray, 
        alpha_gauss=0.005, alpha_overlap=0.025,
        img_nuc_gauss=0, img_nuc_overlap=0,
        border_gauss = 8, border_min=16, border_max=64, algo='FourierShift',
    ):

    #estimate the frames anyway with not yet updated img_nuc
    frame_gauss = frame + 2**13 - img_nuc_gauss
    frame_overlap = frame_gauss + 2**13 - img_nuc_overlap

    # frame_overlap = frame + 2**13 - img_nuc_overlap
    # frame_gauss = frame_overlap + 2**13 - img_nuc_gauss

    # Estimate the motion vector between the previous frame and the current frame both enhanced
    di, dj = motion_estimation_frame(prev_frame=og_frame_n_1, curr_frame=frame, algo=algo)
    # di, dj = motion_estimation_frame(prev_frame=enhanced_frame_n_1, curr_frame=frame_overlap, algo=algo)
    # di, dj = motion_estimation_frame(prev_frame=frame_temp_n_1, curr_frame=frame_overlap, algo=algo)

    # Update threshold from the paper
    vec_size = np.sqrt(di**2 + dj**2)
    update_nuc_gauss = (border_gauss <= vec_size)
    update_nuc_overlap = ((border_min <= vec_size <= border_max) and dj!=0) # need dj translation to remove column noise

    # Perform gaussian mean algorithm on the frame
    if update_nuc_gauss :
        img_nuc_gauss = alpha_gauss * frame + (1 - alpha_gauss) * img_nuc_gauss
        # img_nuc_gauss = alpha_gauss * frame_overlap + (1 - alpha_gauss) * img_nuc_gauss

    # Update the coefficients and estimate the corrected pixel values
    if update_nuc_overlap:        
        img_nuc_overlap = nuc_overlap_frame(frame=frame_gauss, frame_n_1=frame_temp_n_1, di=di, dj=dj, img_nuc=img_nuc_overlap, alphap=alpha_overlap)
        # img_nuc_overlap = nuc_overlap_frame(frame=frame, frame_n_1=og_frame_n_1, di=di, dj=dj, img_nuc=img_nuc_overlap, alpha=alpha_overlap)
        # img_nuc_overlap = nuc_overlap_frame(frame=frame_overlap, frame_n_1=frame_temp_n_1, di=di, dj=dj, img_nuc=img_nuc_overlap, alpha=alpha_overlap)

    return np.where(frame_overlap < 0, 0, frame_overlap).astype(frame.dtype), np.where(frame_gauss < 0, 0, frame_gauss).astype(frame.dtype), img_nuc_gauss.astype(frame.dtype), img_nuc_overlap.astype(frame.dtype)#, update_nuc_gauss, update_nuc_overlap
    # return np.where(frame_gauss < 0, 0, frame_gauss).astype(frame.dtype), np.where(frame_overlap < 0, 0, frame_overlap).astype(frame.dtype), img_nuc_gauss.astype(frame.dtype), img_nuc_overlap.astype(frame.dtype)#, update_nuc_gauss, update_nuc_overlap


def nuc_overlap_frame(
        frame: list | np.ndarray, frame_n_1: list | np.ndarray, 
        di=16, dj=16,
        img_nuc=0, alpha=0.05, 
    ):
    # define overlaping areas
    idi_min_n_1, idi_max_n_1 = max(0,-di), min(frame.shape[0]-1, frame.shape[0]-1 - di)
    jdj_min_n_1, jdj_max_n_1 = max(0,-dj), min(frame.shape[1]-1, frame.shape[1]-1 - dj)    
    
    idi_min, idi_max = max(0,di), min(frame.shape[0]-1, frame.shape[0]-1 + di)
    jdj_min, jdj_max = max(0,dj), min(frame.shape[1]-1, frame.shape[1]-1 + dj)

    # define overlaping area of both error matrix
    idi_full_min, idi_full_max = max(idi_min_n_1, idi_min), min(idi_max_n_1, idi_max)
    jdi_full_min, jdi_full_max = max(jdj_min_n_1, jdj_min), min(jdj_max_n_1, jdj_max)
    
    #get error from one way and then the other way on bot original frames
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
    return img_nuc