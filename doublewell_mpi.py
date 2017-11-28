#
# Samples a 2d double-well distribution using Umbrella sampling in the temperature (parallel version)
#
# Usage:
# > mpirun -np N python gaussian.py
# where "N" is the number of cores to run on, e.g. 4
#

import usample.usample
import numpy as np
import emcee 

#
# Sample a 2D Gaussian
#
# Define the log probability function:
#


def log_prob_fn(p ):
    
    x=p[0]
    y=p[1]
    
    # Two Gaussian distributions, centered on +/- 1 
    lpf = np.log( np.exp(-(x-1)**2) + np.exp(-(x+1)**2  ))
    # And an independent Gaussian in another direction
    lpf -= 0.5*y*y
     
    
    return lpf 

#
# Now create the umbrella sampler object 
us = usample.UmbrellaSampler( log_prob_fn  , mpi=True, debug=True,  burn_acor=20 )

#
# Build umbrellas along a line between two points: the two peaks of the distribution.
# This line is a 1D "CV" or Collective Variable, embedded in the higher dimensions.
cvfn = ["line", [np.array([-1.0,0]), np.array([1.0,0])] ]

# The start of the line is "0", and the end is "1".
# We define the centers of the biasing distribution. Here we define four umbrellas:
centers = np.linspace( 0 , 1 , 4 ) 

# We may also make use of sampling using different temperatures within a window.
# Lets try sampling at a temperature of 1 and 5.
temps = [1,5]  

# Note that this gives a total of 4x2=8 umbrellas overall.
# We are umbrella sampling in two directions:
#   - perpendicular to the 1D line defined in "cvfn" (4 umbrellas)
#   - the temperature  (2 umbrellas)
# Feel free to experiment with the parameters.
#

#
# Other than a "line" in cvfn, we may also use "grid" to place umbrellas in a checkerboard.
# The usage looks like this:
##cvfn = ["grid", [np.array([-1.0,0]), np.array([1.0,0]), np.array([-1.0,1.0])] ]
##centers = [ [ii,jj] for ii in np.linspace( 0 , 1 , 3 ) for jj in np.linspace( 0 , 1 , 5 ) ]
# Where "grid" takes three points, defining the 2D plane and the boundaries.
#

#
# Once we have defined the umbrellas as above, we can add them all together using the following routine: 
us.add_umbrellas( temperatures=temps, centers=centers, cvfn=cvfn , numwalkers=4 , ic=np.array([0,0]) , sampler=emcee.EnsembleSampler )

# 
# Then run for 10000 steps in each window.
# Output stats every [freq] steps
# Try to replica exchange [repex]-many walkers every [freq] steps
#

pos,weights,prob = us.run(10000 , freq=100, repex=0    )

#
# We save the output
#

if (us.is_master() ):
    
    x = np.append( pos , weights , axis=1 )
    x = np.append( x , prob , axis=1 )

    np.savetxt(  'results_doublewell_mpi.txt' , x )



us.close_pools()

