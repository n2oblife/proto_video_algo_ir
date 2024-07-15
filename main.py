from utils.cv2_video_computation import *
from utils.interract import *
from utils.metrics import *
from utils.target import *
from noise_gen import apply_noise
from utils.data_handling import *
from algorithms.morgan import morgan

def apply_nuc_algorithms(
        frames: np.ndarray, 
        algorithms: List[str]=['SBNUCIRFPA'], 
        algo_params: dict = {},
        save_path: str = None
    ) -> dict:
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

    if algorithms == ['all']:
        algorithms =  nuc_algorithms.keys()

    if isinstance(algorithms, str):
        algorithms = [algorithms]

    # Dictionary to store results for each algorithm
    results = {}

    for algo in algorithms:
        if algo in nuc_algorithms:
            if save_path:
                if check_files_exist(save_path, [algo + '.pkl']): 
                    results[algo] = load_data(save_path + '/' + algo + '.pkl')
                else :
                    # Get the parameters for the current algorithm if provided
                    params = algo_params.get(algo, {})
                    # Apply the algorithm to the frames
                    results[algo] = nuc_algorithms[algo](frames, **params)
                    save_frames(results[algo], save_path + '/' + algo + '.pkl')
            else:
                # Get the parameters for the current algorithm if provided
                params = algo_params.get(algo, {})
                # Apply the algorithm to the frames
                results[algo] = nuc_algorithms[algo](frames, **params)
        else:
            print(f"Warning: Algorithm '{algo}' is not recognized and will be skipped.")

    return results


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

###
# CLI
###

# python main.py -p C:/Users/zKanit/Pictures/sbnuc_offset -w 640 -he 480 -d 14b -fps 60 -n 60 --show_video --clean
# python main.py -p C:\Users\zKanit\Pictures\***_no_nuc -save C:\Users\zKanit\Documents\***\proto_video_algo_ir\log -d 14b -nuc all -m all    

if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args()
    stable_frame_number = args['stable_frame']

    # Load video frames based on provided arguments
    if args['save_folder']:
        if check_files_exist(args['save_folder'], ['clean_frames.pkl', 'noisy_frames.pkl', 'noise.pkl']) == {'clean_frames.pkl': True, 'noisy_frames.pkl': True, 'noise.pkl': True}:
            clean_frames, noisy_frames, noise = load_data(args['save_folder'] + '/clean_frames.pkl'), load_data(args['save_folder'] + '/noisy_frames.pkl'), load_data(args['save_folder'] + '/noise.pkl')
            n_to_compute = min(len(clean_frames), args['num_frames'])
        else:
            clean_frames, noisy_frames, n_to_compute, noise = load_all_frames(args)
            save_frames(clean_frames, args['save_folder'] + '/clean_frames.pkl')
            save_frames(noisy_frames, args['save_folder'] + '/noisy_frames.pkl')
            save_frames(noise, args['save_folder'] + '/noise.pkl')
    else:
        clean_frames, noisy_frames, n_to_compute, noise = load_all_frames(args)

    # If the user requested to show the video, display the noisy frames
    if args['show_video']:
        showing_input = input("Which frames to show? \nClean or Noisy or both: c / n / b\n")

        if showing_input == "c":
            print(" --- Showing clean frames --- ")
            show_video(frames=clean_frames, title='clean frames', frame_rate=args['framerate'])
        elif showing_input == "n":
            print(" --- Showing noisy frames --- ")
            show_video(frames=noisy_frames, title='noisy frames', frame_rate=args['framerate'])
        elif showing_input == "b":
            print(" --- Showing clean frames --- ")
            show_video(frames=clean_frames, title='clean frames', frame_rate=args['framerate'])
            print(" --- Showing noisy frames --- ")
            show_video(frames=noisy_frames, title='noisy frames', frame_rate=args['framerate'])
        else:
            print("Invalid input. Please enter 'c' for clean frames or 'n' for noisy frames and 'all' for both of them.")


    # Apply non-uniformity correction (NUC) algorithms to the frames
    estimated_frames = apply_nuc_algorithms(frames=noisy_frames[:n_to_compute],
                                            algorithms=args['nuc_algorithm'],
                                            save_path=args['save_folder'])
    
    
    # Compute specified metrics for the estimated frames compared to the original frames
    if args['metrics']:
        metrics = metrics_estimated(estimated_frames, clean_frames, args['metrics'], args['save_folder'])
        plot_metrics(metrics)

    # If the user requested to show the video, display the estimated frames
    if args['show_video']:
        print(" --- Showing frames estimation --- ")
        showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate'])
        # if n_to_compute == len(noisy_frames):
        #     showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate'])
        # else:
        #     showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate']/4)

    # Indicate the completion of the process
    print("DONE!")
