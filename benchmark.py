import time

import cv2
import dask
import dask.array as da
import numpy as np
import scipy.ndimage as ndimage
from numba import njit

"""
References:
http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
https://docs.dask.org/en/stable/array-overlap.html
https://docs.dask.org/en/stable/scheduling.html
"""


def create_data(shape):
    data = np.random.random(shape)
    return data


def time_function(fn, fn_args, n=1):
    times = []
    for i in range(n):
        start = time.perf_counter()
        _ = fn(**fn_args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def scipy_std(array, filter_width):
    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    http://cpsc.yale.edu/sites/default/files/files/tr222.pdf
    """
    n = filter_width ** 2
    shifted = array - np.mean(array)
    conv_sum = ndimage.convolve(shifted, np.ones((filter_width, filter_width)))
    conv_squared_sum = ndimage.convolve(shifted ** 2, np.ones((filter_width, filter_width)))
    variance = (conv_squared_sum - ((conv_sum ** 2) / n)) / n
    std = np.sqrt(variance)
    return std


def scipy_wallis(array, filter_width):
    std = scipy_std(array, filter_width)
    mean = ndimage.uniform_filter(array, filter_width)

    out_array = (array - mean) / std
    return out_array


def cv2_std(array, filter_width):
    n = filter_width ** 2
    shifted = array - np.mean(array)
    filter_kernel = np.ones((filter_width, filter_width))
    conv_sum = cv2.filter2D(shifted, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)
    conv_squared_sum = cv2.filter2D(shifted ** 2, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)

    variance = (conv_squared_sum - ((conv_sum ** 2) / n)) / n
    std = np.sqrt(variance)
    return std


def cv2_wallis(array, filter_width):
    n = filter_width ** 2
    filter_kernel = np.ones((filter_width, filter_width))

    std = cv2_std(array, filter_width)
    mean = cv2.filter2D(array, -1, filter_kernel, borderType=cv2.BORDER_REFLECT) / n

    out_array = (array - mean) / std
    return out_array


def dask_std(array, filter_width):
    n = filter_width ** 2
    shifted = array - array.mean()
    filter_kernel = np.ones((filter_width, filter_width))
    # da_array = da.from_array(array, chunks=(4, 4))
    conv_sum = shifted.map_overlap(lambda x: ndimage.convolve(x, filter_kernel), depth=filter_width, boundary='reflect')
    conv_squared_sum = (shifted ** 2).map_overlap(lambda x: ndimage.convolve(x, filter_kernel), depth=filter_width,
                                                  boundary='reflect')

    variance = (conv_squared_sum - ((conv_sum ** 2) / n)) / n
    std = np.sqrt(variance)
    return std


def dask_wallis(array, filter_width):
    with dask.config.set(scheduler='threads'):  # change to synchronous for single thread
        da_array = da.from_array(array)

        std = cv2_std(array, filter_width)
        mean = da_array.map_overlap(lambda x: ndimage.uniform_filter(x, filter_width), depth=filter_width,
                                    boundary='reflect')
        out_array = (array - mean) / std
        out_array = out_array.compute()
    return out_array


@njit
def local_mean(array):
    return np.mean(array)


@njit
def local_std(array):
    return np.std(array)


def numba_wallis(array, filter_width):
    std = ndimage.generic_filter(array, local_std, (filter_width, filter_width))
    mean = ndimage.generic_filter(array, local_mean, (filter_width, filter_width))
    out_array = (array - mean) / std
    return out_array


if __name__ == '__main__':
    n_runs = 1
    data_width = 7_000  # rough size of landsat 7 scene (not panchromatic)
    kernel_width = 5

    test_data = create_data((data_width, data_width))
    in_data = {'array': test_data, 'filter_width': kernel_width}

    print('starting scipy')
    exec_time = time_function(scipy_wallis, in_data, n_runs)
    print(f'Scipy took {exec_time:.03f}\n')

    print('starting CV2')
    exec_time = time_function(cv2_wallis, in_data, n_runs)
    print(f'CV2 took {exec_time:.03f}\n')

    print('starting dask')
    exec_time = time_function(dask_wallis, in_data, n_runs)
    print(f'Dask took {exec_time:.03f}\n')

    print('starting numba')
    exec_time = time_function(numba_wallis, in_data, n_runs)
    print(f'numba took {exec_time:.03f}\n')

    print('done!')
