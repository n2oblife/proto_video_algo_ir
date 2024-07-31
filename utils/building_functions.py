from utils.cv2_video_computation import *
from utils.interract import *
from utils.metrics import *
from utils.target import *
from noise_gen import apply_noise
from utils.data_handling import *
from algorithms.morgan import morgan
from itertools import product
from typing import List, Dict, Generator


def apply_nuc_algorithms(
        frames: np.ndarray,
        algorithms: List[str]=['SBNUCIRFPA'],
        test_parameters=False,
        save_path: str = None
    ) -> Generator[Dict[str, Dict[str, float]], None, None]:
    """
    Apply multiple NUC algorithms to a sequence of frames with different parameter combinations and return the results.

    This function applies each NUC algorithm specified in the `algorithms` list to the provided
    frames. The results of each algorithm with different parameter combinations are stored in a dictionary.

    Args:
        frames (np.ndarray): A numpy array of frames (grayscale) from the video.
        algorithms (List[str]): A list of algorithm names to apply.
        test_parameters (bool): Whether to use test parameters.
        save_path (str, optional): Path to save the results. Defaults to None.

    Returns:
        Generator[Dict[str, Dict[str, float]], None, None]: A generator that yields dictionaries where the keys are the algorithm names and the values are dictionaries
                                                            containing the results of applying the corresponding algorithms to the frames
                                                            with different parameter combinations.
    """
    # Build the dictionary of NUC algorithms
    nuc_algorithms = build_nuc_algos()

    if algorithms == ['all']:
        algorithms = list(nuc_algorithms.keys())

    if isinstance(algorithms, str):
        algorithms = [algorithms]

    algo_params = build_parameters(test_parameters, algorithms)

    for algo in algorithms:
        if algo in nuc_algorithms:
            # Get the parameters for the current algorithm if provided or empty dict
            params_dict = algo_params.get(algo, {})

            # Generate all combinations of parameters
            param_combinations = list(product(*params_dict.values()))

            for combination in param_combinations:
                # Create a dictionary of parameters for the current combination
                current_params = dict(zip(params_dict.keys(), combination))

                if save_path:
                    # Create file name from the algo name and parameters names
                    param_str = '_'.join(f"{k}_{v}" for k, v in current_params.items())
                    file_name = f"{algo}_{param_str}.pkl"
                    file_path = os.path.join(save_path, file_name)

                    if check_files_exist(save_path, [file_name]):
                        result = load_data(file_path)
                    else:
                        # Apply the algorithm to the frames with the current parameter combination
                        result = nuc_algorithms[algo](frames, **current_params)
                        save_frames(result, file_path)
                else:
                    # Apply the algorithm to the frames with the current parameter combination
                    result = nuc_algorithms[algo](frames, **current_params)

                yield {algo: result}
        else:
            print(f"Warning: Algorithm '{algo}' is not recognized and will be skipped.")


def load_all_frames(args: dict) -> np.ndarray:
    """
    Load and process frames based on provided arguments.

    Args:
        args (dict): A dictionary of arguments with keys such as 'clean', 'num_frames', 'width', and 'height'.

    Returns:
        tuple: A tuple containing clean frames, noisy frames, the number of frames to compute, and the noise array.
    """
    # Initialize noise to None
    noise = None
    stable_frame_number = args['stable_frame']

    # Check if the 'clean' flag is set in the arguments
    if args['clean']:
        clean_frames = np.array(load_frames(args))
        n_to_compute = min(len(clean_frames), args['num_frames'])
        clean_frames = clean_frames[stable_frame_number:stable_frame_number + n_to_compute]
        noisy_frames, noise = apply_noise(clean_frames, width=args['width'], height=args['height'])
    else:
        # If the 'clean' flag is not set, load noisy frames
        noisy_frames = np.array(load_frames(args))
        n_to_compute = min(len(noisy_frames), args['num_frames'])
        noisy_frames = noisy_frames[stable_frame_number:stable_frame_number + n_to_compute]
        
        # Estimate the clean frames by applying a Gaussian filter to each noisy frame
        # clean_frames = np.array(
        #     [frame_gauss_3x3_filtering(frame) for frame in tqdm(noisy_frames, desc="Estimating clean frame", unit="frame")], 
        #     dtype=noisy_frames.dtype
        # )
        clean_frames = morgan(noisy_frames)

    # Ensure noise is not None; if it is, initialize it with zeros of the same shape as clean_frames
    if noise is None:
        noise = np.zeros_like(clean_frames)

    return clean_frames, noisy_frames, n_to_compute, noise

def build_parameters(test_parameters = False, nuc_algo = []):
    if test_parameters:
        # TODO - revoir les valeurs qui méritent d'etre utilisées
        all_algo_parameters = {
        'SBNUCIRFPA':               {'eta' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'AdaSBNUCIRFPA':            {'K' : [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'AdaSBNUCIRFPA_reg':        {'K' : [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                     'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'CstStatSBNUC':             {'threshold' : [0, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30], # px error threshold - 0 to 255
                                     'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'SBNUCLMS':                 {'K' : [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                     'M' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'threshold' : [0, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30]},  # px error threshold - 0 to 255
        'SBNUCif_reg':              {'lr' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'algo' : ['OptFlow', 'BlockMotion', 'FourierShift']}, # motion estimation
        'AdaSBNUCif_reg':           {'lr' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'algo' : ['OptFlow', 'BlockMotion', 'FourierShift']}, # motion estimation
        # 'CompTempNUC':            {},        
        'NUCnlFilter':              {'Ts' : [0, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30], # px error threshold - 0 to 255
                                     'Tg' : [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 300, 500, 1000, 15000, 2000]}, # number of px error to estimate motion (TODO - change it to frame motion rate ?)
        'RobustNUCIRFPA':           {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'ada' : [True, False],
                                     'algo' : ['OptFlow', 'BlockMotion', 'FourierShift']}, # motion estimation
        'AdaRobustNUCIRFPA':        {'alpha_m' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'ada' : [True, False],
                                     'algo' : ['OptFlow', 'BlockMotion', 'FourierShift']}, # motion estimation
        'SBNUC_smartCam_pipeA':     {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'exp_window' : [True, False]},
        'SBNUC_smartCam_pipeB':     {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'alpha_p' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'exp_window' : [True, False]},
        'SBNUC_smartCam_pipeC':     {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'alpha_p' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'alpha_avg' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'SBNUC_smartCam_own_pipe':  {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'alpha_p' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'exp_window' : [True, False]},
        'SBNUCcomplement':          {}, # no more parameter
        'morgan':                   {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'morgan_moving':            {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'moving_rate' : [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]}, # threshold based on frame motion rate
        'morgan_filt' :             {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'morgan_filt_haut' :        {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'Adamorgan' :               {'K' : [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                     'A' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_NUCnlFilter' :         {}, # no more parameter
        'zac_smartCam' :            {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'alpha_k' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_AdaSBNUCIRFPA_window' :{'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'K' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_AdaSBNUCIRFPA_mean' :  {'eta' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_AdaSBNUCIRFPA_reg' :   {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'K' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_RobustNUCIRFPA' :      {'alpha_m' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_SBNUCcomplement' :     {}, # no more parameter
        'zac_CstStatSBNUC' :        {'threshold' : [0, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30], # px error threshold - 0 to 255
                                     'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        'zac_SBNUCLMS' :            {'K' : [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                     'M' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'threshold' : [0, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30]},  # px error threshold - 0 to 255
        'zac_AdaSBNUCif_reg' :      {'lr' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'algo' : ['OptFlow', 'BlockMotion', 'FourierShift']}, # motion estimation
        'zac_morgan' :              {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'alpha_k' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'moving_rate' : [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]}, # threshold based on frame motion rate
        'zac_Adamorgan' :           {'alpha' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005], 
                                     'K' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005],
                                     'A' : [1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.0000005, 0.0000001, 0.00000005]},
        }
        return {key: all_algo_parameters[key] for key in nuc_algo if key in all_algo_parameters}
    else:
        return {}