# Wallis Filter Benchmarking
Use the `environment.yml` to recreate the conda environment used in this trial.

**Results With Inputs:**  
Image size: 7000x7000  
Kernel size: 5x5  
n Iterations: 5  

Scipy took 2.373 s on average  
CV2 took 0.751 s on average  
Dask took 0.751 s on average  
Julia on auto thread took 19.506 s for single thread, then stalled out
(julia benchmarks performed in ARM64 M1 Mac conda environment which may have impacted performance)  

**Benchmark Computer Specs**  
10 core Apple M1 pro with 32 gb of RAM
