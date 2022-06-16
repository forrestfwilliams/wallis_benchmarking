import scipy.ndimage as ndimage
import numpy as np
import time


def time_function(fn, fn_args, n=1):
    times = []
    for i in range(n):
        start = time.perf_counter()
        fn(**fn_args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def mean1(array, filter_kernel):
    out_array = ndimage.convolve(array, filter_kernel) / np.sum(filter_kernel)
    return out_array


def mean2(array, filter_kernel):
    out_array = ndimage.uniform_filter(array, filter_kernel.shape[0])
    return out_array


# test_data = np.zeros((5, 5))
# test_data[-1:] = 1
# kernel = np.ones((3, 3))

n_runs = 5
data_width = 5000
kernel_width = 5

test_data = np.ones((data_width, data_width))
kernel = np.ones((kernel_width, kernel_width))
in_data = {'array': test_data, 'filter_kernel': kernel}

print(test_data)
filter1 = time_function(mean1, in_data, n_runs)
print(filter1)
filter2 = time_function(mean2, in_data, n_runs)
print(filter2)
