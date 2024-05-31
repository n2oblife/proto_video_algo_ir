from utils.cv2_video_computation import *
from utils.interract import *
from utils.metrics import *
from utils.target import *
from algorithms.AdaSBNUCIRFPA import AdaSBNUCIRFPA, SBNUCIRFPA

def init():
    return

def update():
    return

def estimate():
    return

def metrics():
    return

# python main.py -p C:/Users/zKanit/Pictures/sbnuc_offset -w 640 -he 480 -d 14b -fps 1 -n 60 --show_video
    
if __name__ == "__main__":

    # Help : if pb with cv2 : reshape array and convert dtype to uint8

    set_logging_info()
    args = build_args()
    frames = load_frames(args)
    breakpoint()
    frame_est_SBNUCIRFPA = SBNUCIRFPA(frames[:args['num_frames']])
    frame_target = frame_mean_filtering(frames[args['num_frames']-1])
    # frame_est_AdaSBNUCIRFPA = AdaSBNUCIRFPA(frames[:args['num_frames']])
    if args['show_video']:
        print(" --- Showing estimation --- ")
        show_video(frames=frame_est_SBNUCIRFPA, title="SBNUCIRFPA", frame_rate=args['framerate'])
        # show_video(frames=frame_est_AdaSBNUCIRFPA, title="AdaSBNUCIRFPA", frame_rate=args['framerate'])

    breakpoint()
    metrics = compute_metrics(original_frames=frames_target,
                              enhanced_frames=frame_est_SBNUCIRFPA
                              )

    print(f"metrics : {metrics}")
    print("DONE!")