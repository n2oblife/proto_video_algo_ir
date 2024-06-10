import numpy as np

def init_nuc(image: list|np.ndarray) -> dict[str, np.ndarray]:
    """
    Initialize nuclear matrices for the image processing.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The first array is initialized with ones, having the same shape as the input image.
            - The second array is initialized with zeros, having the same shape as the input image.
    """
    return { "g" : np.ones((len(image), len(image[0])), dtype=image.dtype), 
            "o" : np.zeros((len(image), len(image[0])), dtype=image.dtype)
            }

def process_return(returned, og):
    if isinstance(og, (np.ndarray)):
        return np.array(returned, dtype=og.dtype)
    else:
        return returned

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
    p = 0
    for l in range(i - k_size//2, i + k_size//2+1):
        kernel_im.append([])
        for m in range(j - k_size//2, j + k_size//2+1):
            if 0 <= l < len(image) and 0 <= m < len(image[0]):
                kernel_im[p].append(image[l][m])
            else :
                kernel_im[p].append(None)
        p+=1
    return process_return(returned=rm_None(kernel_im), og=image)


def rm_None(data):
    """
    Recursively remove None elements and empty lists from a deeply nested list.

    Args:
        data (list): The input list, potentially containing nested lists, None elements, and empty lists.

    Returns:
        list: A new list with all None elements and empty lists removed.
    """
    if isinstance(data, list):
        # Recursively process each item in the list, and filter out None and empty lists
        cleaned_data = [rm_None(item) for item in data if item is not None]
        # Filter out empty lists
        return [item for item in cleaned_data if (not isinstance(item, list)) or (len(item) > 0)]
    else:
        return data


def Yij(frame: list | np.ndarray):
    """
    Retrieve the value of the central pixel in a given frame.

    Args:
        frame (list | np.ndarray): The input frame as a 2D list or array.

    Returns:
        The value of the central pixel in the frame.
    """
    return frame[len(frame) // 2][len(frame[0]) // 2]



def compute_error(Xest: float | np.ndarray, target: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the error between the estimated and target values.

    Args:
        Xest (float | np.ndarray): The estimated value(s).
        target (float | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The difference between the estimated and target values.
    """
    return np.abs(Xest - target)

def loss(Xest: list | np.ndarray, target: list | np.ndarray) -> float | np.ndarray:
    """
    Calculate the loss (sum of squared errors) between the estimated and target values.

    Args:
        Xest (list | np.ndarray): The estimated value(s).
        target (list | np.ndarray): The target value(s).

    Returns:
        float | np.ndarray: The sum of squared errors between the estimated and target values.
    """
    return np.sum(compute_error(Xest, target)**2)


def sgd_step(coeff: float | np.ndarray, lr: float | np.ndarray, delta: float | np.ndarray, bias: float | np.ndarray = 0) -> float | np.ndarray:
    """
    Perform a single step of stochastic gradient descent (SGD) to update a coefficient.

    -> updated_coeff = coeff - lr * delta + bias

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

    -> estimated_value = g * y + o + b

    Args:
        g (float | np.ndarray): The first input value or coefficient.
        y (float | np.ndarray): The second input value or coefficient.
        o (float | np.ndarray): The third input value or coefficient.
        b (float | np.ndarray, optional): An optional bias term. Defaults to 0.

    Returns:
        float | np.ndarray: The estimated value resulting from the linear combination of the inputs and the bias.
    """
    return g * y + o + b


def reshape_array(_array: np.ndarray) -> np.ndarray:
    """
    This function checks if the input array is 3-dimensional with a shape of (w, h, 1).
    If so, it removes the singleton dimension and returns a 2D array.
    Otherwise, it returns the original array unchanged.

    Args:
        _array (np.ndarray): The input array, which can be either 2D or 3D.

    Returns:
        np.ndarray: The resized 2D array if the original array had a shape of (w, h, 1),
                    otherwise the original array.
    """
    if _array.ndim == 3 and _array.shape[2] == 1:
        return np.squeeze(_array, axis=-1)
    else:
        return _array
