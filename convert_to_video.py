import numpy as np
import cv2
import os
from utils.interract import set_logging_info, build_args_show_result
from utils.data_handling import load_data
from tqdm import tqdm
from utils.cv2_video_computation import save_video_to_mp4, save_video_to_avi

def get_pkl_files(folder_path):
    """
    Get a list of the names of the .pkl files in the specified folder path.

    Args:
        folder_path (str): The path to the folder containing the .pkl files.

    Returns:
        List[str]: A list of the names of the .pkl files in the folder.
    """
    pkl_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            base_name = os.path.splitext(file_name)[0]
            pkl_files.append(base_name)
    return pkl_files

if __name__ == "__main__":
    # Set up logging with INFO level to capture detailed runtime information
    set_logging_info()

    # Parse command-line arguments to get user inputs
    args = build_args_show_result()

    # Get the list of .pkl files in the specified folder path
    pkl_files = get_pkl_files(args['folder_path'])

    while True:
        print()
        showing_input = input(f"Choose among these computed frames {pkl_files}\n")
        print()

        if showing_input not in pkl_files and showing_input != "all":
            input_break = input("You didn't choose an existing file, do you want to quit? Y/n\n")
            if input_break != "n":
                break
        else:
            try:
                if showing_input == "all":
                    inputs = pkl_files
                else:
                    inputs = [showing_input]
                for inp in inputs:
                    frames = load_data(args['folder_path'] + f'/{inp}.pkl')
                    save_video_to_avi(
                        frames, 
                        output_path=args['folder_path'] + f'/{inp}', 
                        fps=args['framerate'], 
                        title=inp
                        )
            except FileNotFoundError:
                print(f"Sorry the file {showing_input}.pkl was not found.\n Please try again.\n")
