from utils.cv2_video_computation import *
from utils.interract import *
from utils.metrics import *
from utils.target import *
from algorithms.AdaSBNUCIRFPA import *
from algorithms.SBNUCrgGLMS import *
from algorithms.SBNUCif_reg import *

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



# python main.py -p C:/Users/zKanit/Pictures/sbnuc_offset -w 640 -he 480 -d 14b -fps 1 -n 60 --show_video
    
if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args()

    # Load video frames based on provided arguments
    frames = load_frames(args)

    # Apply non-uniformity correction (NUC) algorithms to the frames
    estimated_frames = apply_nuc_algorithms(frames=frames,
                                            algorithms=args['nuc_algorithm'])

    # Apply mean filtering to the target frame for noise reduction
    frame_target = frame_mean_filtering(frames[args['num_frames'] - 1])
    
    # If the user requested to show the video, display the estimated frames
    if args['show_video']:
        print(" --- Showing frames estimation --- ")
        showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate'])
    
    # Compute specified metrics for the estimated frames compared to the original frames
    metrics = metrics_estimated(estimated_frames, frames, args['metrics'])

    # Indicate the completion of the process
    print("DONE!")
