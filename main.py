from utils.cv2_video_computation import *
from utils.interract import *
from utils.metrics import *
from utils.target import *
from algorithms.AdaSBNUCIRFPA import *
from algorithms.SBNUCrgGLMS import *
from algorithms.SBNUCif_reg import *
from algorithms.NUCnlFilter import *
from algorithms.SBNUC_smartCam import *
from algorithms.RobustNUCIRFPA import *
from algorithms.SBNUCcomplement import *
# from algorithms.CompTempNUC import *
from noise_gen import apply_noise

def init():
    return

def update():
    return

def estimate():
    return

def metrics():
    return

def build_parameters(algo_used:list[str]):
    return{

    }

def build_nuc_algos():
    """
    Build a dictionary mapping SBNUC algorithm names to their corresponding functions.

    This function creates and returns a dictionary where the keys are the names of different
    Scene-Based Non-Uniformity Correction (SBNUC) algorithms, and the values are the functions
    that implement these algorithms.

    Returns:
        dict: A dictionary mapping SBNUC algorithm names (str) to their corresponding functions.
              The available algorithms are:
              -> 'SBNUCIRFPA': Function for SBNUCIRFPA algorithm
              -> 'AdaSBNUCIRFPA': Function for adaptive SBNUCIRFPA algorithm
              -> 'AdaSBNUCIRFPA_reg': Function for adaptive SBNUCIRFPA with registration
              -> 'CstStatSBNUC': Function for constant statistics SBNUC
              -> 'SBNUCLMS': Function for SBNUCLMS algorithm
              -> 'SBNUCif_reg': Function for SBNUC with interframe registration
              -> 'AdaSBNUCif_reg': Function for adaptive SBNUC with interframe registration
              -> 'CompTempNUC' : Function to compensate Temperature variations through NUC
              -> 'NUCnlFilter' : Function to apply a non linear filter to the NUC
              -> 'RobustNUCIRFPA' : Function to apply a robust NUC on IRFPA
              -> 'AdaRobustNUCIRFPA' : Function to apply a robust NUC on IRFPA with adaptation
              -> 'SBNUC_smartCam_pipeA' : Function to apply a NUC smart camera algorithm using pipeline A
              -> 'SBNUC_smartCam_pipeB' : Function to apply a NUC smart camera algorithm using pipeline B
              -> 'SBNUC_smartCam_pipeC' : Function to apply a NUC smart camera algorithm using pipeline C
              -> 'SBNUCcomplement' : Function to apply a complement to the first filter
    """
    # TODO add new algos when implementation
    return {
        'SBNUCIRFPA': SBNUCIRFPA,                # Function for SBNUCIRFPA algorithm
        'AdaSBNUCIRFPA': AdaSBNUCIRFPA,          # Function for adaptive SBNUCIRFPA algorithm
        'AdaSBNUCIRFPA_reg': AdaSBNUCIRFPA_reg,  # Function for adaptive SBNUCIRFPA with registration
        'CstStatSBNUC': CstStatSBNUC,            # Function for constant statistics SBNUC
        'SBNUCLMS': SBNUCLMS,                    # Function for SBNUCLMS algorithm
        'SBNUCif_reg': SBNUCif_reg,              # Function for SBNUC with interframe registration
        'AdaSBNUCif_reg': AdaSBNUCif_reg,        # Function for adaptive SBNUC with interframe registration
        # 'CompTempNUC': CompTempNUC,              # Function to compensate Temperature variations through NUC
        'NUCnlFilter': NUCnlFilter,            # Function to apply a non linear filter to the NUC
        'RobustNUCIRFPA': RobustNUCIRFPA,        # Function to apply a robust NUC on IRFPA
        'AdaRobustNUCIRFPA': AdaRobustNUCIRFPA,  # Function to apply a robust NUC on IRFPA with adaptation
        'SBNUC_smartCam_pipeA': SBNUC_smartCam_pipeA,  # Function to apply a NUC smart camera algorithm using pipeline A
        'SBNUC_smartCam_pipeB': SBNUC_smartCam_pipeB,  # Function to apply a NUC smart camera algorithm using pipeline B
        'SBNUC_smartCam_pipeC': SBNUC_smartCam_pipeC,  # Function to apply a NUC smart camera algorithm using pipeline C
        'SBNUCcomplement': SBNUCcomplement       # Function to apply a complement to the first filter
    }


def apply_nuc_algorithms(frames: np.ndarray, algorithms: List[str]=['SBNUCIRFPA'], algo_params: dict = {}) -> dict:
    """
    Apply multiple NUC algorithms to a sequence of frames and return the results.

    This function applies each NUC algorithm specified in the `algorithms` list to the provided
    frames. The results of each algorithm are stored in a dictionary.

    Args:
        frames (np.ndarray): A numpy array of frames (grayscale) from the video.
        algorithms (List[str]): A list of algorithm names to apply. 
        algo_params (dict, optional): A dictionary of additional parameters for the algorithms.
                                      The keys should be algorithm names, and the values should be
                                      dictionaries of parameters for those algorithms. Defaults to {}.

    Returns:
        dict: A dictionary where the keys are the algorithm names and the values are the results
              of applying the corresponding algorithms to the frames.
    """
    # Build the dictionary of NUC algorithms
    nuc_algorithms = build_nuc_algos()

    # Dictionary to store results for each algorithm
    results = {}
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    for algo in algorithms:
        if algo in nuc_algorithms:
            # Get the parameters for the current algorithm if provided
            params = algo_params.get(algo, {})
            # Apply the algorithm to the frames
            results[algo] = nuc_algorithms[algo](frames, **params)
        else:
            print(f"Warning: Algorithm '{algo}' is not recognized and will be skipped.")

    return results



# python main.py -p C:/Users/zKanit/Pictures/sbnuc_offset -w 640 -he 480 -d 14b -fps 60 -n 60 --show_video --clean
    
if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args()
    stable_frame_number = args['stable_frame']

    # Load video frames based on provided arguments, must be clean frames
    if args['clean']:
        clean_frames = np.array(load_frames(args))
        n_to_compute = min(len(clean_frames), args['num_frames'])
        clean_frames = clean_frames[stable_frame_number:stable_frame_number+n_to_compute]
        noisy_frames, noise = apply_noise(clean_frames, widht=args['width'], height=args['height'])
    else:
        noisy_frames = np.array(load_frames(args))
        n_to_compute = min(len(noisy_frames), args['num_frames'])
        noisy_frames = noisy_frames[stable_frame_number:stable_frame_number+n_to_compute]
        # clean_frames = np.array([frame_gauss_3x3_filtering(frame) for frame in tqdm(noisy_frames, desc="Estimating clean frame", unit="frame")], dtype=noisy_frames.dtype)
    
    # If the user requested to show the video, display the noisy frames
    # if args['show_video']:
    #     print(" --- Showing noisy frames --- ")
    #     show_video(frames=noisy_frames, title='noisy frames', frame_rate=args['framerate'])

    # Apply non-uniformity correction (NUC) algorithms to the frames
    estimated_frames = apply_nuc_algorithms(frames=noisy_frames[:n_to_compute],
                                            algorithms=args['nuc_algorithm'])
    
    # If the user requested to show the video, display the estimated frames
    if args['show_video']:
        print(" --- Showing frames estimation --- ")
        if n_to_compute == len(noisy_frames):
            showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate'])
        else:
            showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate']/4)
    
    # Compute specified metrics for the estimated frames compared to the original frames
    metrics = metrics_estimated(estimated_frames, clean_frames, args['metrics'])

    plot_metrics(metrics)

    # Indicate the completion of the process
    print("DONE!")
