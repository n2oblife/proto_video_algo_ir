# Modeling and compensating temperature dependent NUC noise in IR microbolometer cameras
# CompTempNUC

import numpy as np
from tqdm import tqdm
from utils.common import *
from utils.target import *
from utils.Mavlink_temp import *
import yaml

def parse_config(file_path):
    """
    Parse the YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

def build_S_params(config: dict, gain=None) -> tuple[float, float, float]:
    """
    Build S_l config of the paper.

    Args:
        config (dict): Configuration dictionary containing parameters.
        gain (float, optional): Gain value, currently not used.

    Returns:
        tuple[float, float, float]: Computed S0, S1, S2 parameters.
    """
    # TODO: Check how to get the gain
    # Estimate the parameters by performing three calibration pixelwise at three different temperatures:
    # S0*Yk - Si-S2*dTk = Xk, where k is for one of the three temperatures.
    # every coeff should be a pixelwise value
    S0 = config['beta'] * config['G'] / (1 * config['R0'] * config['alpha']) # the one must be replaced by the pixel gain
    S1 = config['beta'] * config['G'] / config['alpha']
    S2 = config['beta'] * config['G']
    return S0, S1, S2

def estimate_frame(frame, dT, dT_h, S0, S1, S2):
    """
    Estimate the X frame.

    Args:
        frame (np.ndarray): Input frame.
        dT (float): Temperature difference.
        dT_h (float): Temperature correction term.
        S0 (float): S0 parameter.
        S1 (float): S1 parameter.
        S2 (float): S2 parameter.

    Returns:
        np.ndarray: Estimated X frame.
    """
    return S0 * frame + S1 + S2 * dT + dT_h

def estimate_dT(frame_n_1, dT_n_1, eta=0.0012, k_size=3):
    """
    Estimate the temperature difference (dT).

    Args:
        frame_n_1 (np.ndarray): Previous frame.
        dT_n_1 (float): Previous temperature difference.
        eta (float, optional): Learning rate for temperature estimation. Defaults to 0.0012.
        k_size (int, optional): Kernel size for mean filtering. Defaults to 3.

    Returns:
        float: Updated temperature difference (dT).
    """
    return dT_n_1 + 2 * eta * np.sum(frame_n_1 - frame_mean_filtering(frame_n_1, k_size))

def CompTempNUC(frames: list | np.ndarray, dT_n_1, eta=0.0012, k_size=3):
    """
    Apply the CompTempNUC algorithm to a sequence of frames.

    Args:
        frames (list | np.ndarray): Input frames.
        dT_n_1 (float): Initial temperature difference.
        eta (float, optional): Learning rate for temperature estimation. Defaults to 0.0012.
        k_size (int, optional): Kernel size for mean filtering. Defaults to 3.

    Returns:
        np.ndarray: Estimated frames after applying CompTempNUC.
    """
    # Load configuration from YAML file
    conf_path = 'C:\\Users\\zKanit\\Documents\\Bertin_local\\proto_video_algo_ir\\algorithms\\CompTempNUC_conf.yaml'
    config = parse_config(conf_path)
    all_frames_est = []
    gain = np.zeros(frames[0], dtype=frames.dtype)  # Initialize gain array
    frame_n_1 = frames[0]  # Set the first frame as the initial previous frame

    # Build the S parameters using the configuration
    S0, S1, S2 = build_S_params(config)
    for frame in tqdm(frames[1:], desc="CompTempNUC algorithm", unit="frame"):
        # Estimate the frame and temperature difference for the current frame
        frame_est, dT = CompTempNUC_frame(frame, frame_n_1, dT_n_1, dT_h, S0, S1, S2, k_size, eta)
        all_frames_est.append(frame_est)  # Append the estimated frame to the list
        frame_n_1 = frame  # Update the previous frame for the next iteration

    return np.array(all_frames_est, dtype=frames[0].dtype)

def CompTempNUC_frame(
        frame: list | np.ndarray, 
        frame_n_1: list | np.ndarray, 
        dT_n_1, 
        dT_h,
        S0, S1, S2,
        k_size=3,
        eta=0.0012):
    """
    Process a single frame using the CompTempNUC algorithm.

    Args:
        frame (np.ndarray): Input frame.
        frame_n_1 (np.ndarray): Previous frame.
        dT_n_1 (float): Previous temperature difference.
        dT_h (float): Temperature correction term.
        S0 (float): S0 parameter.
        S1 (float): S1 parameter.
        S2 (float): S2 parameter.
        k_size (int, optional): Kernel size for mean filtering. Defaults to 3.
        eta (float, optional): Learning rate for temperature estimation. Defaults to 0.0012.

    Returns:
        tuple: (Estimated frame, updated temperature difference)
    """
    dT = estimate_dT(frame_n_1, dT_n_1, eta, k_size)  # Estimate the new temperature difference
    est_frame = estimate_frame(frame, dT, dT_h, S0, S1, S2)  # Estimate the current frame
    return est_frame, dT  # Return the estimated frame and updated temperature difference
