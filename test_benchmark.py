import pytest
import benchmark
import numpy as np


def generate_data():
    image = np.ones((5, 5)) * 20
    image[:2, :2] += np.ones((2, 2)) * 30
    return image


def test_scipy_std_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    assert np.round(std, 5) == np.round(benchmark.scipy_std(image, width)[2, 2], 5)


def test_scipy_wallis_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    mean = np.mean(subset.flatten())
    wallis = (image[2, 2] - mean) / std
    assert np.round(wallis, 5) == np.round(benchmark.scipy_wallis(image, width)[2, 2], 5)


def test_cv2_std_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    assert np.round(std, 5) == np.round(benchmark.cv2_std(image, width)[2, 2], 5)


def test_cv2_wallis_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    mean = np.mean(subset.flatten())
    wallis = (image[2, 2] - mean) / std
    assert np.round(wallis, 5) == np.round(benchmark.cv2_wallis(image, width)[2, 2], 5)


def test_dask_std_value():
    import dask
    import dask.array as da
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())

    with dask.config.set(scheduler='threads'):  # change to sychronous for single thread
        da_image = da.from_array(image)
        da_std = benchmark.dask_std(da_image, width)
        da_std.compute()

    assert np.round(std, 5) == np.round(da_std[2, 2], 5)


def test_dask_wallis_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    mean = np.mean(subset.flatten())
    wallis = (image[2, 2] - mean) / std
    assert np.round(wallis, 5) == np.round(benchmark.dask_wallis(image, width)[2, 2], 5)


def test_numba_wallis_value():
    width = 3
    image = generate_data()
    subset = image[1:4, 1:4]
    std = np.std(subset.flatten())
    mean = np.mean(subset.flatten())
    wallis = (image[2, 2] - mean) / std
    assert np.round(wallis, 5) == np.round(benchmark.numba_wallis(image, width)[2, 2], 5)
