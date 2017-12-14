Umbrella Sampling python package `USample`
=====

A lightweight python implementation of the umbrella sampling method for use in efficient calculation of tail probabilities in various data science applications.

Companion article: Umbrella sampling: a powerful method to sample tails of distributions  
  Charles Matthews, Jonathan Weare, Andrey Kravtsov and Elise Jennings  
  Available here: <https://arxiv.org/> (link forthcoming)

### Using the package

There are several examples included to use the package: sampling a Gaussian distribution, a double-well multimodal distribution, and a cosmological example. They can be run as:   
`python gaussian.py` for the 2D Gaussian example  
`mpirun python gaussian_mpi.py` to run the sampler in parallel, if MPI4PY is available.

Example of how to sample a 2D Gaussian pdf is also available in 'gauss_us_example.ipynb' Jupyter notebook.

The package is easy to set up and run, it requires only the sampler (emcee is the preferred choice), the log posterior function and an initial condition. See the example files for snippets on how to set up the sampler.

Example of how to sample posterior of the matter and vacuum density + nuisance parameters from supernovae type Ia JLA v3 sample is provided in jla_like_emcee.py, where emcee sampler (without umbrella sampling) is used to sample the likelihood. 

Before running this code you need to download SNIa data and pre-computed luminaosity distance table packaged into a single archive. The data can be downloaded by running 'python download_sn_data.py datadir' where is directory into which you want to unpack the data. The code will download zip archive and will unpack it into subdirectory sn_data within the specified directory sn_data. Once data is downloaded run the code as 'python jla_like_emcee.py datadir/sn_data'
This code will produce two data files: 

Example jla_like_us.py is a similar code, but which now uses umbrella sampling with temperature stratification using emcee sampler to sample distributions within each window. <tt>mpi4py</tt> is required to run this code and the supernova data, as described above.
Run the code as `mpirun python jla_like_us.py datadir`, where datafir is path to the directory with downloaded JLA SN data. 



### License

The code is free and available to all under the GNU GPL3 license.
