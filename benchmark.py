import numpy as np
import scipy.ndimage as ndimage
import cv2
import dask.array as da
import dask
import time

'''
References:
http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
https://docs.dask.org/en/stable/array-overlap.html
https://docs.dask.org/en/stable/scheduling.html
'''


def create_data(shape):
    data = np.random.random(shape)
    return data


def time_function(fn, fn_args, n=1):
    times = []
    for i in range(n):
        start = time.perf_counter()
        fn(**fn_args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def scipy_wallis(array, filter_kernel):
    kernel_sum = np.sum(filter_kernel)
    mean = ndimage.uniform_filter(array, kernel.shape[0])
    squared_mean = ndimage.uniform_filter(mean ** 2, kernel.shape[0])
    std = np.sqrt((squared_mean - mean) ** 2) * np.sqrt(kernel_sum / (kernel_sum - 1.0))

    out_array = (array - mean) / std
    return out_array


def dask_wallis(array, filter_kernel):
    with dask.config.set(scheduler='thread'): #change to sychronous for single thread
        kernel_sum = np.sum(filter_kernel)
        da_array = da.from_array(array, chunks=(4, 4))
        mean = da_array.map_overlap(lambda x: ndimage.uniform_filter(x, kernel.shape[0]), depth=kernel.shape[0],
                                    boundary='reflect')
        squared_mean = mean ** 2
        squared_mean = squared_mean.map_overlap(lambda x: ndimage.uniform_filter(x, kernel.shape[0]), depth=kernel.shape[0],
                                                boundary='reflect')

        std = np.sqrt((squared_mean - mean) ** 2) * np.sqrt(kernel_sum / (kernel_sum - 1.0))

        out_array = (array - mean) / std
        out_array = out_array.comput()
    return out_array


def cv2_wallis(array, filter_kernel):
    kernel_sum = np.sum(filter_kernel)
    mean = cv2.filter2D(array, -1, filter_kernel, borderType=cv2.BORDER_CONSTANT) / kernel_sum
    squared_mean = cv2.filter2D(mean ** 2, -1, kernel, borderType=cv2.BORDER_CONSTANT) / kernel_sum

    std = np.sqrt((squared_mean - mean) ** 2) * np.sqrt(kernel_sum / (kernel_sum - 1.0))

    out_array = (array - mean) / std
    return out_array


n_runs = 5
data_width = 7_000  # rough size of landsat 7 scene (not panchromatic)
kernel_width = 5

test_data = create_data((data_width, data_width))
kernel = np.ones((kernel_width, kernel_width))
in_data = {'array': test_data, 'filter_kernel': kernel}

print('starting scipy')
exec_time = time_function(scipy_wallis, in_data, n_runs)
print(f'Scipy took {exec_time:.03f}\n')

print('starting CV2')
exec_time = time_function(cv2_wallis, in_data, n_runs)
print(f'CV2 took {exec_time:.03f}\n')

print('starting dask')
exec_time = time_function(cv2_wallis, in_data, n_runs)
print(f'Dask took {exec_time:.03f}\n')

print('done!')
