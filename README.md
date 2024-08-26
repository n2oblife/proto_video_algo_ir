# proto_video_algo_ir

This repo's goal is to found a suitable SBNUC for IR camera.
This script tests the SBNUC (Scene Based Non Uniformity Correction) algorithm on IR (Infrared) camera data to evaluate its efficiency. The SBNUC algorithm is designed to address various types of defective effect in an image, such as noisy pixels, sun ghosting etc.
Let's dive into how to use this library or add complementary algorithms.

## How to use it 

It is a standard python repository. Set a virtual environnement and install the dependencies using the 'requirement.txt' file.
Launch the main.py file and see the helper to know how to use it on any video or binary files.

### Install working environnment
Check that you have Python3 installed :

```shell
python --version
```

Otherwise, [install it](https://www.python.org/downloads/).

Then go at the root of the folder where the `main.py` lies. Set up a virtual environnment dedicated to this project to install all libraries whitout twinkering with the global environnment :

```shell
python -m venv env
python install -r requirements.txt
```

Then activate the virtual environnment :

```shell
env\Scripts\activate
```

### Usage 

To use this script, run the following command:

```shell
python main.py -h
```

This will display the help message with all available options.

#### Required Arguments

- `-p` or `--folder_path`: The path to the folder containing all the binary files of the video.

#### Optional Arguments

- `-nuc` or `--nuc_algorithm`: Algorithms to use for nuc adaptation. You can specify multiple algorithms. The default algorithm is `morgan_overlap`.
- `-save` or `--save_folder`: The path to the folder where the results from this algorithm will be saved.
- `-video` or `--save_video`: Flag to save the video files into `.avi` files. This requires the `-save` option to be set.
- `-w` or `--width`: The width of the video. The default width is `640`.
- `-he` or `--height`: The height of the video. The default height is `480`.
- `-d` or `--depth`: The bits' depth of the video. The default depth is `14b`.
- `-fps` or `--framerate`: The framerate per second for displaying the video. The default framerate is `30`.
- `-s` or `--show_video`: Show the video if this flag is set. The default is `False`.
- `-k` or `--kernel_size`: The size of the kernel to use for filtering. Must be an odd number. The default kernel size is `3`.
- `-m` or `--metrics`: Metrics to compute. You can specify multiple metrics.
- `-t` or `--test_parameters`: Flag to test multiple parameters values. The default is `False`.

#### Example Usage

To test the `morgan_overlap` algorithm on a video with a framerate of 30 and save the results to a folder, and save the video in the saving path, run the following command:

```shell
python main.py -p /path/to/video/files -nuc morgan_overlap -save /path/to/save/folder -video -fps 30
```

To test multiple algorithms and compute multiple metrics, run the following command:

```shell
python main.py -p /path/to/video/files -nuc morgan_overlap SBNUC_smartCam_pipeA -m mse psnr -save /path/to/save/folder
```

To test multiple parameters values (these testing parameters can be changed in `utils/building_functions.py` in the `building_parameters()` function), run the following command:

```shell
python main.py -p /path/to/video/files -nuc morgan_overlap -t
```

## Add algorithm

To implement algorithm just add a file in the `algorithms` folder with a function using as primary argument `frames`, a list of 2D numpy arrays and do whatever you want with it as long as you return a list of the new frames :

```python
import numpy as np
from tqdm import tqdm

def new_algo(frames: list | np.ndarray, **other_parameters):
    all_frames_est = []
    img_nuc = np.full(shape=frames[0].shape, dtype=frames[0].dtype, fill_value=2**13)

    # Use tqdm to show progress while iterating through frames
    for frame in tqdm(frames[], desc="new algo processing", unit="frame"):
        #...
        frame_estimated = #...
        #...

        all_frames_est.append(frame_est)
    return np.array(all_frames_est, dtype=frames[0].dtype)
```

Then add the new python file name inside `algorithm/__init__.py` file (with all the other files), and your algorithm name inside `utils/interract` - `build_nuc_algo()` function with this syntax :

```python
def build_nuc_algos():
    return {
        # ...
        'morgan'            : alg.morgan.morgan,
        'morgan_overlap'    : alg.morgan.morgan_overlap,
        'cli_name_of_aglo'  : alg.new_file.new_algo,
        # ...
    }
```

## Complementary elements

This library also holds a lot of tools to develop a new algorithm very easily, with filter, metrics, data handling functions etc.

### Metrics

To add new metrics just go into the `utils/metrics.py` file and add a new metric function using an enhanced frame and/or comparing frame and return a number whatever the type :

```python
def new_metric(original:np.ndarray, enhanced:np.ndarray)->float:
    return np.mean((original - enhanced))

def new_metric2(enhanced:np.ndarray)->int:
    return np.mean((original**2))  
```

Then add you metric name as shown in the first function of the file :

```python
def init_metrics()->dict[str, list]:
    return {
        "mse"               : [],
        "psnr"              : [],
        "new_metric_name"   : [],
        }
```

### Target function (filters)

Already got a lot of filter functions available. Filter take arrays as arguments and return arrays.
To have a low computation time, use as much as possible numpy and all the dedicated functions.

### Motion estimation

Two functions to use and which work well : 
- `motion/motion_estimation.py` - `di, dj = motion_estimation_frame(previous_frame, current_frame)` : uses FFT to measure vectors of motion
- `algorithms/NUCnlFilter` - `boolean_motion =  M_n(current_frame, previous_frame, pix_change_threshold, nbre_px_thr)` : compares frames pixel by pixel and based on the pixel values change and the number of changing pixels considers if there is motion or not (watch out for temporal noise - should use a moving rate instead of a fixed number (`nbre_px_thr = frame.shape[0] * frame.shape[1] * moving_rate`)