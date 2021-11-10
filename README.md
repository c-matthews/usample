Umbrella Sampling python package `USample`
=====

A lightweight Python implementation of the umbrella sampling method for use in efficient calculation of tail probabilities in various data science applications. The code can be used in both Python 2.7 and Python 3.x.

Companion article: Umbrella sampling: a powerful method to sample tails of distributions  
  Charles Matthews, Jonathan Weare, Andrey Kravtsov and Elise Jennings  
  https://arxiv.org/abs/1712.05024

### Using the package

Installation is via the setup.py file.

The package requires only the sampler, the log posterior function and an initial condition to use. See the example files for snippets on how to set up the sampler object.

Examples included here all rely on the <a href="http://dfm.io/emcee/current/">emcee</a> sampler by <a href="https://arxiv.org/abs/1202.3665">Foreman-Mackey et al. (2013)</a> that implements the affine-invariant ensemble sampling method of <a href="">Goodman & Weare (2010)</a>. See the emcee code <a href="http://dfm.io/emcee/current/">website</a> for instructions for how to install it. However, other MCMC samplers can be substituted instead by users.

There are several examples included to use the package: sampling a Gaussian distribution, a double-well multimodal distribution, and a cosmological example. They can be run as `python gaussian.py` for the 2D Gaussian example  
`mpirun python gaussian_mpi.py` to run the sampler in parallel, if MPI4PY is available.
Example of how to sample a 2D Gaussian pdf is also available in <a href="https://github.com/c-matthews/usample/blob/master/gauss_us_example.ipynb">`gauss_us_example.ipynb`</a> Jupyter notebook.

Example of how to sample posterior of the matter and vacuum density + nuisance parameters from supernovae type Ia JLA v3 sample with umbrella sampling is provided in <a href="https://github.com/c-matthews/usample/blob/master/jla_like_us.py">jla_like_us.py</a>. This example uses umbrella sampling with temperature stratification using emcee sampler to sample distributions within each window. <tt>mpi4py</tt> is required to run this code and the supernova data, as described above.

Before running this example code you need to download SNIa data and pre-computed luminosity distance table packaged into a single archive. The data can be downloaded by running 'python download_sn_data.py datadir' where datadir is directory into which you want to unpack the data. The code will download zip archive and will unpack it into subdirectory sn_data within the specified directory sn_data.

Once data is downloaded, run the code as `mpirun python jla_like_us.py datadir`, where datafir is path to the directory with downloaded JLA SN data.

This code will produce three data files, `pos_nielsen_3par_us.npy`, `weights_nielsen_3par_us.npy` and `prob_nielsen_3par_us.npy` containing walker chains in the 3d parameter space (10 parameters of the 13-parameter likelihood are kept fixed in this example), US weights associated with each position, and values of the log-likelihood at each position.  Notebook <a href="https://github.com/c-matthews/usample/blob/master/jla_like_check_chain.ipynb">`jla_like_test_chain.ipynb`</a> shows how to read the data files and plot posterior distribution of the mean matter and vacuum densities.

The likelihood in `jla_like_us.py` was used to produce Figure 5 of our paper. It is based on the likelihood of <a href="http://adsabs.harvard.edu/abs/2016NatSR...635596N">Nielsen et al. (2016)</a>, but adds two parameters, `eta_c` and `eta_x`, to model the evolution of the mean color correction (`eta_c`) and stretch (`eta_x`). We caution the users that this likelihood is included for illustration only, as it does not model the selection effects of the supernovae samples, and thus should not be used for production cosmological constraints.

### License

The code is free and available to all under the GNU GPL3 license.
