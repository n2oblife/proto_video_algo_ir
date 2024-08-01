from utils.interract import set_logging_info, build_args_show_result
from utils.cv2_video_computation import show_video
from utils.data_handling import load_data
import os

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
        showing_input = input(f"Choose among these computed frames {pkl_files}\n")
        
        if showing_input not in pkl_files:
            input_break = input("You didn't choose an existing file, do you want to quit ? Y/n\n")
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
    