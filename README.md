Umbrella Sampling package `USample` for Python 2.7
=====

A lightweight python implementation of the umbrella sampling method for use in efficient calculation of tail probabilities in various data science applications.

Companion article: Umbrella sampling: a powerful method to sample tails of distributions  
  Charles Matthews, Jonathan Weare, Andrey Kravtsov and Elise Jennings  
  Available here: <https://arxiv.org/> (link forthcoming)

### Using the package

There are several examples included to use the package: sampling a Gaussian distribution, a double-well multimodal distribution, and a cosmological example. They can be run as:   
`python gaussian.py` for the 2D Gaussian example  
`mpirun python gaussian_mpi.py` to run the sampler in parallel, if MPI4PY is available.

The package is easy to set up and run, it requires only the sampler (emcee is the preferred choice), the log posterior function and an initial condition. See the example files for snippets on how to set up the sampler.

### License

The code is free and available to all under the GNU GPL3 license.
