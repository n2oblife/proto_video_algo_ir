from utils.interract import set_logging_info, build_args_show_result
from utils.cv2_video_computation import show_video
from utils.data_handling import load_data
from main import build_nuc_algos

if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args_show_result()

    algos = [algo for algo in build_nuc_algos().keys()] 

    while True:
        showing_input = input(f"Choose among these computed frames {algos}\n")
        
        if showing_input not in algos:
            input_break = input("You didn't choose an existing algorithm, do you want to quit ? Y/n\n")
            if input_break != "n":
                break
        else :
            try:
                frames = load_data(args['folder_path'] + f'/{showing_input}.pkl')
                print(f" --- Showing {showing_input} frames --- ")
                show_video(frames=frames, 
                        title=showing_input,
                        frame_rate=args['framerate'])
            except FileNotFoundError:
                print(f"Sorry the file {showing_input}.pkl was not found.\n Please try again.\n")
    