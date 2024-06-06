import numpy as np
from utils.common import *
from utils.cv2_video_computation import *

def gen_noise_list(clean_frames, mu=0, sig=10):
    noisy_frames = []
    # generating the mean of the column noise
    noise_col_mean = np.random.normal(loc=mu, scale=np.sqrt(sig), size=clean_frames[0][0])
    for frame in tqdm(clean_frames, desc="Adding noise to clean video", unit="frame"):
        noisy_frames.append(gen_noise_frame(frame, noise_col_mean, mu, sig))
    return

def gen_noise_frame(frame, noise_col_mean, mu=0, sig=10):

    for j in range(len(frame[0])):
        col_noise = np.random.normal(loc=noise_col_mean[j], scale=np.sqrt(sig/2), size=len(frame))
        for i in range(len(frame)):
            frame[i][j] = frame[i][j] + col_noise[j]
    return frame
