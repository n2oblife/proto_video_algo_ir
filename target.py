from common import build_kernel
import cv2
import numpy as np
from statistics import mean 
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.signal import wiener
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma

def mean_filter(image:list, i:int, j:int, k_size=3)->float:
    """
    Apply a mean filter to a specific pixel in an image.

    Args:
        image (list): The input image as a 2D list or array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        float: The mean value of the kernel surrounding the target pixel.
    """
    kernel_im = []
    for l in range(i-k_size, i+k_size):
        for m in range(j-k_size, j+k_size):
            if 0 <= l < len(image) and 0 <= m < len(image[0]):
                kernel_im.append(image[l, m])
    return mean(kernel_im)

def gauss(x, sig=1):
    """
    Calculate the Gaussian function value for a given x.

    Args:
        x (float): The input value.
        sig (int, optional): The standard deviation of the Gaussian function. Defaults to 1.

    Returns:
        float: The Gaussian function value.
    """
    return 1/(2*np.pi*sig**2) * np.exp(-x**2 / (2*sig**2))

def gauss_kernel(x, y, sig=1):
    """
    Calculate the Gaussian function value for a given x and y.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        sig (int, optional): The standard deviation of the Gaussian function. Defaults to 1.

    Returns:
        float: The Gaussian function value.
    """
    return 1/(2*np.pi*sig**2) * np.exp(-(x**2 + y**2) / (2*sig**2))

def gaussian_filtering(image:np.ndarray, i:int, j:int, sig=1., k_size=3):
    """
    Apply a Gaussian filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The standard deviation for the Gaussian filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return gaussian_filter(kernel_im, sig)

def uniform_filtering(image:np.ndarray, i:int, j:int, sig=1., k_size=3):
    """
    Apply a uniform filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The size of the uniform filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return uniform_filter(kernel_im, sig)

def median_filtering(image:np.ndarray, i:int, j:int, sig=1., k_size=3)->float:
    """
    Apply a median filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The size of the median filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        float: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return median_filter(kernel_im, sig)

def bilateral_filtering(image:np.ndarray, i:int, j:int, d=9, sigmaColor=75, sigmaSpace=75, k_size=3)->cv2.Mat:
    """
    Apply a bilateral filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        d (int, optional): Diameter of each pixel neighborhood. Defaults to 9.
        sigmaColor (int, optional): Filter sigma in the color space. Defaults to 75.
        sigmaSpace (int, optional): Filter sigma in the coordinate space. Defaults to 75.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        cv2.Mat: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return cv2.bilateralFilter(kernel_im, d, sigmaColor, sigmaSpace)

def wiener_filtering(image:np.ndarray, i:int, j:int, k_size=3):
    """
    Apply a Wiener filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return wiener(kernel_im)

def bilateral_denoise_filtering(image:np.ndarray, i:int, j:int, k_size=3, sigma_color=0.05, sigma_spatial=15, multichannel=False):
    """
    Apply a bilateral denoise filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.
        sigma_color (float, optional): Filter sigma in the color space. Defaults to 0.05.
        sigma_spatial (int, optional): Filter sigma in the coordinate space. Defaults to 15.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    return denoise_bilateral(kernel_im, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=multichannel)

def nl_means_filtering(image:np.ndarray, i:int, j:int, k_size=3, h=1.15, fast_mode=True, patch_size=5, patch_distance=6, multichannel=False):
    """
    Apply a non-local means filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.
        h (float, optional): The parameter regulating filter strength. Defaults to 1.15.
        fast_mode (bool, optional): If True, a fast version of the algorithm is used. Defaults to True.
        patch_size (int, optional): The size of the patches used for comparison. Defaults to 5.
        patch_distance (int, optional): The maximum distance in pixels between patches. Defaults to 6.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    kernel_im = build_kernel(image, i, j, k_size)
    sigma_est = np.mean(estimate_sigma(kernel_im, multichannel=multichannel))
    return denoise_nl_means(kernel_im, h=h*sigma_est, fast_mode=fast_mode, patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)
