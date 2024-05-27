import cv2 
import numpy as np
from tqdm import tqdm


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

def show_video(frames: list, title='frames', frame_rate=30) -> None:
    """
    Display video frames in a window from a list.

    Args:
        frames (list): A list containing the frames of the video.
        title (str, optional): The title of the window. Defaults to 'frames'.
        frame_rate (int, optional): The frame rate for displaying the video. Defaults to 30.
    """
    # Display each frame 
    for frame in frames:
        # Display the resulting frame 
        cv2.imshow(title, frame)
        # Press 'q' on keyboard to exit 
        if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
            break

    # Close all the frames 
    cv2.destroyAllWindows()
