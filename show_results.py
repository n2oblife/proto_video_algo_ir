from utils.interract import set_logging_info, build_args_show_result
from utils.cv2_video_computation import show_video
from utils.data_handling import load_data

if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args_show_result()

    algos = [   
        'clean_frames',
        'noisy_frames',
        'SBNUCIRFPA',
        'AdaSBNUCIRFPA',
        'AdaSBNUCIRFPA_reg',
        'CstStatSBNUC',
        'SBNUCLMS',
        'SBNUCif_reg',
        'AdaSBNUCif_reg',
        'RobustNUCIRFPA',
        'AdaRobustNUCIRFPA',
        'SBNUC_smartCam_pipeA',
        'SBNUC_smartCam_pipeB',
        'SBNUC_smartCam_pipeC',
        'SBNUCcomplement'
    ]

    while True:
        showing_input = input(f"Choose among these computed frames {algos}\n")
        
        if showing_input not in algos:
            input_break = input("You didn't choose an existing algorithm, do you want to quit ? Y/n\n")
            if input_break != "n":
                break
        else :
            frames = load_data(args['folder_path'] + f'/{showing_input}.pkl')
            print(f" --- Showing {showing_input} frames --- ")
            show_video(frames=frames, 
                    title=showing_input,
                    frame_rate=args['framerate'])
    