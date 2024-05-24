import numpy as np

def init_nuc(image: list|np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize nuclear matrices for the image processing.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The first array is initialized with ones, having the same shape as the input image.
            - The second array is initialized with zeros, having the same shape as the input image.
    """
    return np.ones((len(image), len(image[0]))), np.zeros((len(image), len(image[0])))

def build_kernel(image: list|np.ndarray, i: int, j: int, k_size=3) -> list|np.ndarray:
    """
    Extract a kernel (submatrix) from an image centered around a specific pixel.

    Args:
        image (list): The input image as a 2D list or array.
        i (int): The row index of the target pixel around which the kernel is built.
        j (int): The column index of the target pixel around which the kernel is built.
        k_size (int, optional): The size of the kernel (half-width). Defaults to 3.

    Returns:
        list: A flattened list containing the values of the kernel surrounding the target pixel.
    """
    kernel_im = []
    for l in range(i - k_size, i + k_size):
        for m in range(j - k_size, j + k_size):
            if 0 <= l < len(image) and 0 <= m < len(image[0]):
                kernel_im.append(image[l][m])
            else :
                kernel_im.append(None)
    if type(image) == np.array:
        return np.array(kernel_im)
    else:
        return kernel_im
    
def Yij(frame: list | np.ndarray):
    """
    Retrieve the value of the central pixel in a given frame.

    Args:
        frame (list | np.ndarray): The input frame as a 2D list or array.

    Returns:
        The value of the central pixel in the frame.
    """
    return frame[len(frame) // 2][len(frame[0]) // 2]



def error(Xest: float | np.ndarray, target: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the error between the estimated and target values.

    Args:
        Xest (float | np.ndarray): The estimated value(s).
        target (float | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The difference between the estimated and target values.
    """
    return Xest - target

def loss(Xest: list | np.ndarray, target: list | np.ndarray) -> float | np.ndarray:
    """
    Calculate the loss (sum of squared errors) between the estimated and target values.

    Args:
        Xest (list | np.ndarray): The estimated value(s).
        target (list | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The sum of squared errors between the estimated and target values.
    """
    if isinstance(Xest, np.ndarray):
        return np.sum(error(Xest, target)**2)
    else:
        return sum([error(Xest[i], target[i])**2 for i in range(len(Xest))])

def sgd_step(coeff: float | np.ndarray, lr: float | np.ndarray, delta: float | np.ndarray, bias: float | np.ndarray = 0) -> float | np.ndarray:
    """
    Perform a single step of stochastic gradient descent (SGD) to update a coefficient.

    Args:
        coeff (float | np.ndarray): The current value of the coefficient.
        lr (float | np.ndarray): The learning rate for the update step.
        delta (float | np.ndarray): The gradient of the loss function with respect to the coefficient.
        bias (float | np.ndarray, optional): An optional bias term to be added to the update. Defaults to 0.

    Returns:
        float | np.ndarray: The updated value of the coefficient after the SGD step.
    """
    return coeff - lr * delta + bias

def Xest(g: float | np.ndarray, y: float | np.ndarray, o: float | np.ndarray, b: float | np.ndarray = 0.) -> float | np.ndarray:
    """
    Estimate a value using a linear combination of inputs and an optional bias.

    Args:
        g (float | np.ndarray): The first input value or coefficient.
        y (float | np.ndarray): The second input value or coefficient.
        o (float | np.ndarray): The third input value or coefficient.
        b (float | np.ndarray, optional): An optional bias term. Defaults to 0.

    Returns:
        float | np.ndarray: The estimated value resulting from the linear combination of the inputs and the bias.
    """
    return g * y + o + b
