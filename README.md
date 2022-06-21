# Wallis Filter Benchmarking

## Environment setup
A conda `environment.yml` is provided to recreate the conda environment used in this trial.
Once you have [conda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
create and activate the environment with:
```shell
conda env update -f environment.yml
conda activate wallis
```

*Note: you can re-run the above commands at anytime to keep your environment up to date.*

### Additional Julia setup

Before you can run the julia benchmarks, you'll need to install the `ImageFiltering` Julia package:
```shell
julia -e 'import Pkg; Pkg.add("ImageFiltering")'
```

## Run the benchmarks

Run the Python benchmarks with:
```shell
python benchmark.py
```

Run the Julia benchmarks with:
```shell
julia --threads=auto wallis_filter.jl
```

## Results

The  Inputs:
* Image size: 7000x7000
* Kernel size: 5x5
* `n` Iterations: 5

produced the following Benchmark results.

### 10 core Apple M1 pro with 32 gb of RAM

On average:
* Scipy took 2.464 s
* CV2 took 0.726 s
* Dask took 3.640 s
* Numba took 41.844 s
* Julia on auto thread:
  * single thread took 19.506 s
  * runs with multiple threads then stalled out
    (`osx_arm64` conda environments are still experimental and may have impacted performance)

### 12 core Dell XPS 145 with 32 gb of RAM

On average:
* Scipy took 3.860 s
* CV2 took 1.684 s
* Dask took 3.532 s
* Numba took 39.379 s
* Julia on auto thread:
  * single thread took 23.264 s
  * multiple threads took 11.412 s
  * multiple threads with `mapwindow` took 11.890 s
  * multiple threads with `imfilter` took 2.347 s
