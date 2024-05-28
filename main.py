from cv2_video_computation import *
from interract import *

def init():
    return

def update():
    return

def estimate():
    return

def compute_error():
    return

def metrics():
    return

def load_frames(show=False):
    parser_options = []

    parser_options.append(ParserOptions(
        long="folder_path",
        short="p",
        help="The path to the folder containing all the binary files of the video",
        required=True
    ))

    parser_options.append(ParserOptions(
        long="width",
        short="w",
        type=int,
        default=640,
        help="The width of the video",
        required=True
    ))

    parser_options.append(ParserOptions(
        long="height",
        short="he",
        type=int,
        default=480,
        help="The height of the video",
        required=True
    ))

    parser_options.append(ParserOptions(
        long="depth",
        short="d",
        type=str,
        default="8b",
        help="The bits' depth of video",
        choices=['8b', '14b', '16b', '32b', '64b', '128b', '256b'],
        required=True
    ))

    # TODO add algorithm options

    args = parse_input(
        parser_config=parser_options,
        prog_name="IR SBNUC prototypes"
    )

    print("Input parsed")

    frames = store_video_from_bin(
        folder_path=args['folder_path'],
        width=args['width'],
        height=args['height'],
        channels=1,
        depth=args['depth']
    )

    print(f"{len(frames)} frames stored")

    if show:
        show_video(frames, equalize=True, frame_rate=60)
    return frames
    
if __name__ == "__main__":
    set_logging_info()
    frames = load_frames(show=True)
