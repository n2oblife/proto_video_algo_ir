
###########################################################
# CLI
###########################################################

# python main.py -p C:/Users/zKanit/Pictures/sbnuc_offset -w 640 -he 480 -d 14b -fps 60 -n 60 --show_video --clean
# python main.py -p C:\Users\zKanit\Pictures\***_no_nuc -save C:\Users\zKanit\Documents\***\proto_video_algo_ir\log -d 14b -nuc all -m all -s -t    

###########################################################

from utils.building_functions import *
from utils.interract import set_logging_info, build_args
from utils.metrics import metrics_estimated, plot_metrics
from utils.cv2_video_computation import showing_all_estimated, show_video


if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args()

    # Load video frames based on provided arguments
    if args['save_folder']:

        # create folder if it doesn't exist
        if not os.path.exists(args['save_folder']):
            print(f"Folder {args['save_folder']} does not exist")
            os.makedirs(args['save_folder'])
            print(f"Folder {args['save_folder']} created")

        if check_files_exist(args['save_folder'], ['clean_frames.pkl', 'noisy_frames.pkl', 'noise.pkl']) == {'clean_frames.pkl': True, 'noisy_frames.pkl': True, 'noise.pkl': True}:
            clean_frames, noisy_frames, noise = load_data(args['save_folder'] + '/clean_frames.pkl'), load_data(args['save_folder'] + '/noisy_frames.pkl'), load_data(args['save_folder'] + '/noise.pkl')
            n_to_compute = min(len(clean_frames), args['num_frames'])
        else:
            clean_frames, noisy_frames, n_to_compute, noise = load_all_frames(args)
            save_frames(clean_frames, args['save_folder'] + '/clean_frames.pkl')
            save_frames(noisy_frames, args['save_folder'] + '/noisy_frames.pkl')
            save_frames(noise, args['save_folder'] + '/noise.pkl')
        
        if args['save_video']:
            save_video_to_avi(frames=clean_frames, output_path=args['save_folder'] + '/clean_frames.avi', fps=args['framerate'], title='clean_frames')
            save_video_to_avi(frames=noisy_frames, output_path=args['save_folder'] + '/noisy_frames.avi', fps=args['framerate'], title='noisy_frames')
            save_video_to_avi(frames=noise, output_path=args['save_folder'] + '/noise.avi', fps=args['framerate'], title='noise')
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
    estimated_frames = {}
    for estimated in apply_nuc_algorithms(frames=noisy_frames[:n_to_compute],
                                            algorithms=args['nuc_algorithm'],
                                            save_path=args['save_folder'],
                                            test_parameters=args['test_parameters'],
                                            save_video=args['save_video'],
                                            fps=args['framerate']
                                            ) :
        # stores the estimated frames only if useful after
        if args['metrics'] or args['show_video']:
            estimated_frames[estimated[0]] = estimated[1]
    
    # Compute specified metrics for the estimated frames compared to the original frames
    if args['metrics']:
        metrics = metrics_estimated(estimated_frames, clean_frames, args['metrics'], args['save_folder'])
        plot_metrics(metrics)

    # If the user requested to show the video, display the estimated frames
    if args['show_video']:
        print(" --- Showing frames estimation --- ")
        showing_all_estimated(estimated_frames=estimated_frames, framerate=args['framerate'])

    # Indicate the completion of the process
    print("DONE!")
