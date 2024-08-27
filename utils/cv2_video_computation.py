import os
import cv2
import numpy as np
from tqdm import tqdm


from functools import wraps
import threading
import sys, warnings, logging
import itertools
import time
# from utils.interract import spinner_decorator


def read_bin_file(file_path: str, width: int, height: int, channels: int = 3, depth: str = "8b") -> np.ndarray:
    """
    Read a .bin video file and convert it to an image frame.

    Args:
        file_path (str): The path to the .bin file.
        width (int): The width of the image.
        height (int): The height of the image.
        channels (int): The number of color channels. Default is 3 (RGB).
        bits (str): The bit depth of the image data. Default is "8b".

    Returns:
        np.ndarray: The image frame as a NumPy array.
    """
    # Determine the numpy dtype and byte size based on the bit depth
    dtype_map = {
        '8b': (np.uint8, 1),
        '14b': (np.uint16, 2),
        '16b': (np.uint16, 2),
        '32b': (np.uint32, 4),
        '64b': (np.uint64, 8),
        '128b': (np.void, 16),
        '256b': (np.void, 32)
    }

    if depth not in dtype_map:
        raise ValueError(f"Unsupported bit depth: {depth}\n Supported bit depths are: {list(dtype_map.keys())}")

    dtype, byte_size = dtype_map[depth]
    
    # Calculate the expected size of the binary file in bytes
    frame_size = width * height * channels * byte_size

    # Read the binary data
    with open(file_path, 'rb') as f:
        frame_data = f.read(frame_size)

    # Convert the binary data to a NumPy array
    frame = np.frombuffer(frame_data, dtype=dtype)

    # Reshape the array to the desired image shape
    frame = frame.reshape((height, width, channels))

    # Handle specific bit depths with masking or other processing if needed
    if depth == '14b':
        # Mask the higher 2 bits to ensure 14-bit data
        frame = frame & 0x3FFF

    return frame


def store_video_from_bin(folder_path, width, height, channels=3, depth = "8b")->list[np.ndarray]:
    """
    Store frames from .bin files in a folder into a list.

    Args:
        folder_path (str): The path to the folder containing .bin files.
        width (int): The width of the images.
        height (int): The height of the images.
        channels (int): The number of color channels. Default is 3 (RGB).
        depth (str): bits depth of the pixels. Choices : '8b', '14b', '16b', '32b', '64b', '128b', '256b'. Default is '8b'.

    Returns:
        list: A list containing the frames of the video.
    """
    frames = []

    # List all .bin files in the folder
    bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
    
    # Sort files to maintain the order
    bin_files.sort()

    for bin_file in bin_files:
        file_path = os.path.join(folder_path, bin_file)
        frame = read_bin_file(file_path, width, height, channels, depth)
        frames.append(np.squeeze(frame))
        
    return np.array(frames, dtype=frames[0].dtype)


def store_video(file_path: str = None) -> list:
    """
    Store frames from a video file into a list.

    Args:
        file_path (str): The path to the video file or None for live feed.

    Returns:
        list: A list containing the frames of the video.
    """
    # Create a VideoCapture object and read from input file or live feed
    if not file_path:
        cap = cv2.VideoCapture(0)
        live = True
    else:
        cap = cv2.VideoCapture(file_path)
        live = False

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error opening video file or camera")
        return []

    frames = []

    if live:
        print("Press 'q' to stop recording live feed.")
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                cv2.imshow('Live Feed', frame)
                # Press 'q' on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Reading video frames", unit="frame"):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break


def create_bits_lut(og='14b', target='8b'):
    """
    Create a lookup table (LUT) to map <og> bits values to <target> bits values.
    Supported bits values are : '8b', '14b', '16b', '32b', '64b', '128b', '256b'

    Args:
        og (str, optional): Original bit depth. Defaults to '14b'.
        target (str, optional): Target bit depth. Defaults to '8b'.

    Returns:
        np.ndarray: The lookup table for converting original bit depth to target bit depth.
    """
    # Determine the numpy dtype based on the bit depth
    dtype_map = {
        '8b': np.uint8,
        '14b': np.uint16,
        '16b': np.uint16,
        '32b': np.uint32,
        '64b': np.uint64,
        '128b': np.void,
        '256b': np.void
    }

    # Ensure correct handling of the function
    if (og not in dtype_map) or (target not in dtype_map):
        raise ValueError(f"Unsupported bit depth: {og} or {target}\n Supported bit depths are: {list(dtype_map.keys())}")

    # Transforms the bits depth to actual integers
    og_int, tgt_int = int(og[:-1]), int(target[:-1])

    # Maps the LUT on the correct range of the og bits depth to the targeted bits depth
    lut = np.zeros((2**og_int,), dtype=dtype_map[target])
    for i in range(2**og_int):
        if og_int > tgt_int:
            lut[i] = (i >> (og_int - tgt_int)) & (2**tgt_int - 1)  # Scale down to target-bit range
        else:
            lut[i] = (i << (tgt_int - og_int)) & (2**tgt_int - 1)  # Scale up to target-bit range
    return lut


def create_lut_from_frame(frame, target='8b'):
    """
    Create a lookup table (LUT) from a frame based on histogram equalization.
    Supported bits values are : '8b', '14b', '16b', '32b', '64b', '128b', '256b'

    Args:
        frame (np.ndarray): The input frame from which to create the LUT.
        target (str): The target bit depth. Defaults to '8b'.

    Returns:
        np.ndarray: The lookup table for converting the original frame values to the target bit depth.
    """
    # Determine the numpy dtype based on the bit depth
    dtype_map = {
        '8b': np.uint8,
        '14b': np.uint16,
        '16b': np.uint16,
        '32b': np.uint32,
        '64b': np.uint64,
        '128b': np.void,
        '256b': np.void
    }

    # Ensure the target bit depth is supported
    if target not in dtype_map:
        raise ValueError(f"Unsupported bit depth: {target}\nSupported bit depths are: {list(dtype_map.keys())}")

    # Determine the maximum value for the target bit depth
    target_max_val = 2**int(target[:-1]) - 1

    # Compute the histogram of the input frame
    hist, _ = np.histogram(frame.flatten(), bins=2**(frame.dtype.itemsize*8), range=[0, 2**(frame.dtype.itemsize*8) - 1])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = (cdf.max() - cdf) * target_max_val / (cdf.max() - cdf.min())
    cdf_normalized = np.ma.filled(np.ma.masked_equal(cdf_normalized, 0), 0).astype(dtype_map[target])

    return cdf_normalized

def apply_lut(frame, lut):
    if type(frame) == np.ndarray and type(lut) == np.ndarray:
        return lut[frame]
    else :
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                frame[i][j] = lut[frame[i][j]]
        return frame


def animate(stop_event, messages):
    """Animate a spinner in the console with dynamic loading text.

    Args:
    - stop_event (threading.Event): Event to signal when to stop the spinner.
    - messages (list): List of messages to display in sequence along with the spinner.
    """
    for message, c in zip(itertools.cycle(messages), itertools.cycle(['|', '/', '-', '\\'])):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{message} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\r{message} -> Done! \n')

def spinner_decorator(messages):
    """Decorator to show a spinner with dynamic loading text while a function is running.

    Args:
        messages (list): List of messages to display in sequence along with the spinner.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an event to control the spinner
            stop_event = threading.Event()
            # Start spinner thread
            t = threading.Thread(target=animate, args=(stop_event, messages))
            t.start()
            try:
                # Run the actual function
                result = func(*args, **kwargs)
            finally:
                # Stop the spinner
                stop_event.set()
                # Ensure the spinner thread finishes
                t.join()
            return result
        return wrapper
    return decorator
############################

@spinner_decorator(["showing frames"])
def show_video(frames:list|np.ndarray|cv2.Mat, title='frames', frame_rate=30, equalize=True) -> None:
    """
    Display video frames in a window from a list.

    Args:
        frames (list|np.ndarray|cv2.Mat): A list containing the frames of the video.
        title (str, optional): The title of the window. Defaults to 'frames'.
        frame_rate (int, optional): The frame rate for displaying the video. Defaults to 30.
        equalize (bool, optional): Whether to apply histogram equalization using LUT. Defaults to True.
    """
    # i=0
    # Display each frame
    for frame in frames:
        # i+=1
        # Apply LUT if lut is provided
        if equalize:
            lut = create_lut_from_frame(frame=frame, target='8b')
            frame = apply_lut(frame, lut)
        # Display the resulting frame
        cv2.imshow(title, frame)
        # Press 'q' on keyboard to exit
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break
        # Press 'a' on keyboard to print current frame number
        # if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('a'):
        #     print(f"\nCurrent frame is {i}th")

    # Close all the frames
    cv2.destroyAllWindows()

def show_frame(frame, title='frames', equalize=True):
    # Apply LUT if lut is provided
    if equalize:
        lut = create_lut_from_frame(frame=frame, target='8b')
        frame = apply_lut(frame, lut)
    # Display the resulting frame
    cv2.imshow(title, frame)

def showing_all_estimated(estimated_frames, framerate):
    for algo in estimated_frames:
        print(f" --- Showing {algo} frames --- ")
        show_video(frames=estimated_frames[algo], 
                   title=algo,
                   frame_rate=framerate)

def load_frames(args: dict) -> list:
    """
    Load frames from a binary video file and optionally display them.

    This function reads frames from a binary video file specified by the arguments,
    stores them in a list, and optionally displays the video.

    Args:
        args (dict): A dictionary containing the following keys:

            -> 'folder_path' (str): The path to the folder containing the binary video file.

            -> 'width' (int): The width of each frame in the video.

            -> 'height' (int): The height of each frame in the video.

            -> 'depth' (int): The bit depth of the video.

            -> 'show_video' (bool): A flag indicating whether to display the video after loading.

    Returns:
        list: A list of frames loaded from the binary video file.
    """
    # Load frames from the binary video file using the provided parameters
    frames = store_video_from_bin(
        folder_path=args['folder_path'],
        width=args['width'],
        height=args['height'],
        channels=1,
        depth=args['depth']
    )

    # Print the number of frames that were loaded and stored
    print(f"{len(frames)} frames stored")

    # Return the list of frames
    return frames


def save_video_to_mp4(frames, output_path, fps=30, title = 'frames', bit_depth=8):
    """
    Save a video in a NumPy array to an MP4 file using OpenCV.

    Args:
        frames (np.ndarray): The video as a NumPy array with shape (num_frames, height, width).
        output_path (str): The path to save the output MP4 video.
        fps (int, optional): The frames per second of the output video. Defaults to 30.
        bit_depth (int, optional): The bit depth of the output video. Defaults to 8.
    """
    # Ensure the frames array has the correct shape
    assert frames.ndim == 3, "The frames array should have shape (num_frames, height, width)"

    # Add the channel dimension to the frames array
    frames = np.expand_dims(frames, axis=-1)

    # Get the height, width, and number of channels from the frames array
    _, height, width, _ = frames.shape

    # Create a VideoWriter object to write the video to a file
    out = cv2.VideoWriter(output_path+f'_8b_{int(fps)}fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

    # Write each frame to the video
    for frame in tqdm(frames, desc=f"Saving {title} to MP4", unit="frame"):
        lut = create_lut_from_frame(frame=frame, target='8b')
        frame = apply_lut(frame, lut)
        out.write(frame)

    # Release the VideoWriter object to close the file
    print(f"{title} saved in : {output_path}_8b_{int(fps)}fps.mp4")
    out.release()

def save_video_to_avi(frames, output_path, fps=30, title='frames', bit_depth=8):
    """
    Save a video in a NumPy array to an AVI file using OpenCV.

    Args:
        frames (np.ndarray): The video as a NumPy array with shape (num_frames, height, width).
        output_path (str): The path to save the output AVI video.
        fps (int, optional): The frames per second of the output video. Defaults to 30.
        title (str, optional): The title of the video. Defaults to 'frames'.
        bit_depth (int, optional): The bit depth of the output video. Defaults to 8.
    """
    # Ensure the frames array has the correct shape
    assert frames.ndim == 3, "The frames array should have shape (num_frames, height, width)"

    # Add the channel dimension to the frames array
    frames = np.expand_dims(frames, axis=-1)

    # Get the height, width, and number of channels from the frames array
    _, height, width, _ = frames.shape

    # Create a VideoWriter object to write the video to a file
    # out = cv2.VideoWriter(output_path + f'_8b_{int(fps)}fps.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)

    # Write each frame to the video
    for frame in tqdm(frames, desc=f"Saving {title} to AVI", unit="frame"):
        lut = create_lut_from_frame(frame=frame, target='8b')
        frame = apply_lut(frame, lut)
        out.write(frame)

    # Release the VideoWriter object to close the file
    print(f"{title} saved in: {output_path}")
    out.release()
