import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from algorithms.morgan import morgan_frame
from algorithms.NUCnlFilter import NUCnlFilter_frame_array, M_n

# use NUCnlFilter M_n function to estimate motion
def zac(frames: list | np.ndarray):
    all_frames_est = []      # List to store estimated (corrected) frames
    img_nuc = np.zeros(frames[0].shape)     # Initialize coefficients based on the first frame
    coeffs = init_nuc(frames[0])
    frame_est_n_1 = frames[0]
    Eij_n = np.zeros(frames[0].shape, dtype=frames[0].dtype)

    for i in tqdm(range(1,len(frames[1:])-1), desc="zac algorithm", unit="frame"):
        frame_est, img_nuc = morgan_frame(frames[i], img_nuc)
        if M_n(frame=frame_est, frame_n_1=frame_est_n_1):
            frame_est , coeffs['g'], coeffs['o'], Eij_n= NUCnlFilter_frame_array(
                frame=frames[i], 
                coeffs=coeffs, 
                Eij_n_1=Eij_n,
                N=i
                )        
        else:
            frame_est = Xest(g=coeffs['g'], y=frame_est, o=coeffs['o'])
        all_frames_est.append(frame_est)
        frame_est_n_1 = frame_est
    return np.array(all_frames_est, dtype=frames[0].dtype)
