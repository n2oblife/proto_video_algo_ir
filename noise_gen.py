import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from tqdm import tqdm
from utils.common import *
from utils.cv2_video_computation import *
from utils.target import *

def gen_noise_list(clean_frames:np.ndarray, mu=7, sig=1.1):
    """
    Generate a list of noisy frames by adding Gaussian noise to clean frames.

    Args:
        clean_frames (np.ndarray): Array of clean frames.
        mu (float): Mean of the Gaussian noise. Defaults to 7.
        sig (float): Variance of the Gaussian noise. Defaults to 1.1.

    Returns:
        np.ndarray: Array of noisy frames.
    """
    noisy_frames = []
    noise_frame = gen_noise(len(clean_frames[0][0]), len(clean_frames[0]), mu, sig)
    for frame in tqdm(clean_frames, desc="Adding noise to clean video", unit="frame"):
        noisy_frames.append(frame + noise_frame)
    return np.array(noisy_frames, dtype=clean_frames.dtype)

def gen_noise(width, height, mu=7, sig=1.1):
    """
    Generate a noise frame with column-based Gaussian noise.

    Args:
        width (int): Width of the frame.
        height (int): Height of the frame.
        mu (float): Mean of the Gaussian noise. Defaults to 7.
        sig (float): Variance of the Gaussian noise. Defaults to 1.1.

    Returns:
        np.ndarray: Noise frame.
    """
    # Generating the mean of the column noise
    noise_col_mean = np.random.normal(loc=mu, scale=np.sqrt(sig), size=width)
    return [np.random.normal(loc=col_mean, scale=np.sqrt(sig/5), size=height) for col_mean in noise_col_mean]

# TODO: Implement generating ghosting artifacts
def gen_ghosting(width, height, mu=7, sig=1.1):
    return

def characterize_noise(noisy_frames):
    """
    Characterize the noise in a series of frames.

    Args:
        noisy_frames (np.ndarray): Array of noisy frames.

    Returns:
        None
    """
    # Get a stable scene
    stable_scene = noisy_frames[2990:3231]

    # Compute statistics
    temp_noise = temporal_noise_analysis(stable_scene, False)
    temp_noise_stats = calculate_statistics(temp_noise)
    space_noise = spatial_noise_analysis(stable_scene)
    space_noise_stats = calculate_statistics(space_noise)
    fft_analysis = frequency_domain_analysis(stable_scene)
    fft_noise_stats = calculate_statistics(fft_analysis)
    col_deviation, col_stats = compute_column_statistics(stable_scene)
    col_deviation_stats = calculate_statistics(col_deviation)

    # Plot results
    plot_statistics(temp_noise_stats)
    plot_statistics(col_deviation_stats)
    plot_statistics(space_noise_stats)
    plot_statistics(fft_noise_stats)
    plot_noise_maps(
        np.mean(temp_noise, axis=0), 
        np.mean(space_noise, axis=0), 
        np.mean(fft_analysis, axis=0)
    )
    plot_column_statistics(col_stats)

    # Compute the heatmap
    heatmap = sum_frames_to_heatmap(stable_scene)
    temp_noise_heatmap = sum_frames_to_heatmap(temp_noise)
    space_noise_heatmap = sum_frames_to_heatmap(space_noise)
    fft_heatmap = sum_frames_to_heatmap(fft_analysis)
    col_deviation_heatmap = sum_frames_to_heatmap(col_deviation)

    # Plot heatmap
    plot_heatmap(heatmap)
    plot_heatmap(temp_noise_heatmap)
    plot_heatmap(col_deviation_heatmap)
    plot_heatmap(space_noise_heatmap)
    plot_heatmap(fft_heatmap)

    return None

def calculate_statistics(frames):
    """
    Calculate basic statistics for each frame.

    Args:
        frames (list of np.ndarray): List of frames.

    Returns:
        dict: Dictionary with mean, variance, and standard deviation for each frame.
    """
    stats = {'mean': [], 'variance': [], 'std_dev': []}
    for frame in tqdm(frames, desc="Computing statistics", unit="frame"):
        stats['mean'].append(np.mean(frame))
        stats['variance'].append(np.var(frame))
        stats['std_dev'].append(np.std(frame))
    return stats

def temporal_noise_analysis(frames, meaning=True):
    """
    Analyze temporal noise characteristics.

    Args:
        frames (list of np.ndarray): List of frames.
        meaning (bool): If True, compute the mean of the differences. Defaults to True.

    Returns:
        np.ndarray: Temporal noise map.
    """
    # Compute the difference between consecutive frames
    diff_frames = [np.abs(frames[i] - frames[i-1]) for i in tqdm(range(1, len(frames)), desc="Temporal analysis", unit="frame")]
    if meaning:
        diff_frames = np.mean(diff_frames, axis=0)
    return diff_frames

def spatial_noise_analysis(frames):
    """
    Analyze spatial noise characteristics.

    Args:
        frames (np.ndarray): List of frames.

    Returns:
        np.ndarray: List of spatial noise maps.
    """
    space_noise = []
    for frame in tqdm(frames, desc="Spatial noise analysis", unit="frame"):
        # Use high-pass filter or subtract smoothed frame to highlight noise
        smoothed_frame = frame_gauss_3x3_filtering(frame)
        spatial_noise = frame - smoothed_frame
        space_noise.append(spatial_noise)
    return space_noise

def frequency_domain_analysis(frames):
    """
    Perform frequency domain analysis using Fourier Transform.

    Args:
        frames (np.ndarray): List of frames.

    Returns:
        np.ndarray: Power spectral density of the frames.
    """
    fft_analysis = []
    for frame in tqdm(frames, desc="Frequency domain analysis", unit="frame"):
        f_transform = fft2(frame)
        f_shifted = fftshift(f_transform)
        power_spectrum = np.abs(f_shifted)**2
        fft_analysis.append(power_spectrum)
    return fft_analysis

def plot_statistics(stats):
    """
    Plot statistics.

    Args:
        stats (dict): Dictionary with mean, variance, and standard deviation.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(stats['mean'])
    plt.title('Mean')

    plt.subplot(1, 3, 2)
    plt.plot(stats['variance'])
    plt.title('Variance')

    plt.subplot(1, 3, 3)
    plt.plot(stats['std_dev'])
    plt.title('Standard Deviation')
    
    plt.show()

def plot_noise_maps(temporal_noise, spatial_noise, power_spectrum):
    """
    Plot noise maps.

    Args:
        temporal_noise (np.ndarray): Temporal noise map.
        spatial_noise (np.ndarray): Spatial noise map.
        power_spectrum (np.ndarray): Power spectral density.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(temporal_noise, cmap='hot')
    plt.title('Temporal Noise')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(spatial_noise, cmap='hot')
    plt.title('Spatial Noise')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.log(power_spectrum), cmap='hot')
    plt.title('Power Spectrum (Log Scale)')
    plt.colorbar()

    plt.show()

def compute_column_statistics(frames):
    """
    Compute column statistics (mean, variance, standard deviation) for a sequence of frames.

    Args:
        frames (list of np.ndarray): List of frames.

    Returns:
        dict: Dictionary containing mean, variance, and standard deviation for each column.
    """
    print("Computing column noise stats")
    frames_array = np.array(frames)

    # Compute column means
    column_means = np.mean(frames_array, axis=(0, 1))
    
    # Compute column deviations
    column_deviations = frames_array - column_means[None, None, :]

    # Compute column variances and standard deviations
    column_variances = np.var(column_deviations, axis=(0, 1))
    column_std_devs = np.std(column_deviations, axis=(0, 1))

    column_stats = {
        'mean': column_means,
        'variance': column_variances,
        'std_dev': column_std_devs
    }
    
    return column_deviations, column_stats

def plot_column_statistics(column_stats):
    """
    Plot column statistics.

    Args:
        column_stats (dict): Dictionary containing mean, variance, and standard deviation for each column.
    """
    columns = np.arange(len(column_stats['mean']))
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(columns, column_stats['mean'])
    plt.title('Column Mean')
    plt.xlabel('Column Index')
    plt.ylabel('Mean Pixel Value')

    plt.subplot(1, 3, 2)
    plt.plot(columns, column_stats['variance'])
    plt.title('Column Variance')
    plt.xlabel('Column Index')
    plt.ylabel('Variance')

    plt.subplot(1, 3, 3)
    plt.plot(columns, column_stats['std_dev'])
    plt.title('Column Standard Deviation')
    plt.xlabel('Column Index')
    plt.ylabel('Standard Deviation')

    plt.tight_layout()
    plt.show()

def sum_frames_to_heatmap(frames):
    """
    Sum a list of frames into a single heatmap.

    Args:
        frames (list of np.ndarray): List of frames.

    Returns:
        np.ndarray: Heatmap representing the sum of all frames.
    """
    # Ensure all frames have the same shape
    frame_shape = frames[0].shape
    for frame in frames:
        assert frame.shape == frame_shape, "All frames must have the same shape"
    
    # Initialize an array to store the summed values
    heatmap = np.zeros(frame_shape, dtype=np.float64)
    
    # Sum the frames
    for frame in tqdm(frames, desc="Summing frames"):
        heatmap += frame
    
    return heatmap

def plot_heatmap(heatmap, cmap='hot'):
    """
    Plot a heatmap using Matplotlib.

    Args:
        heatmap (np.ndarray): Heatmap data.
        cmap (str, optional): Colormap to use. Defaults to 'hot'.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap=cmap)
    plt.colorbar()
    plt.title("Summed Frames Heatmap")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

if __name__ == '__main__':
    # Load frames from the binary video file using the provided parameters
    folder_path = 'C:/Users/zKanit/Pictures/sbnuc_offset'
    frames = store_video_from_bin(
        folder_path=folder_path,
        width=640,
        height=480,
        channels=1,
        depth='14b'
    )
    characterize_noise(frames)
