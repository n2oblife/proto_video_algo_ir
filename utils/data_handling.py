import os
import numpy as np
import pickle

def save_frames(frames, filename:str):
    """
    Save a list of NumPy frames to a file.

    Args:
        frames (list of np.ndarray): List of NumPy frames.
        filename (str): The filename to save the data to.
    """
    # Ensure frames are converted to a NumPy array
    if isinstance(frames, list):
        frames = np.array(frames)

    # Save the data to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump(frames, file)
    print(f"Frames saved to {filename}")


def save_dict(dict_list, filename:str):
    """
    Save a dictionary to a file.

    Args:
        dict_list (list of dict): List of dictionaries.
        filename (str): The filename to save the data to.
    """
    # Save the data to a file using pickle
    with open(filename, 'wb') as file:
        pickle.dump(dict_list, file)
    print(f"Dictionary saved to {filename}")


def load_data(filename):
    """
    Load a list of NumPy frames and dictionaries from a file.

    Args:
        filename (str): The filename to load the data from.

    Returns:
        tuple: A tuple containing the list of NumPy frames and the list of dictionaries.
    """
    # Load the data from the file using pickle
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def check_files_exist(folder_path: str, filenames: list[str]) -> dict[str, bool]:
    """
    Check if specified files exist in the given folder.

    Args:
        folder_path (str): The path to the folder.
        filenames (list[str]): List of filenames to check.

    Returns:
        dict[str, bool]: A dictionary indicating the existence of each file.
    """
    # Initialize a dictionary to store the existence status of each file
    files_exist = {}
    
    # Iterate over the list of filenames
    for filename in filenames:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file exists and store the result in the dictionary
        files_exist[filename] = os.path.isfile(file_path)
    
    if len(files_exist) == 1:
        return files_exist[filenames[0]]    
    else:
        return files_exist
    
def check_folder_exists(folder_path: str) -> bool:
    """
    Check if the specified folder exists.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    # Check if the folder exists
    return os.path.isdir(folder_path)