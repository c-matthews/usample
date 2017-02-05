#
# Samples a 2d Gaussian using Umbrella sampling in the temperature (parallel version)
#
# Usage:
# > mpirun -np N python gaussian.py
# where "N" is the number of cores to run on, e.g. 4
#

import usample.usample
import usample.makecv
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
    
    lpf = -(x*x-1)*(x*x-1) - 0.5*y*y
     
    
    return lpf 


cvfn = usample.Makecvfn( [-3,0] , [3,0] ).getcv
 

#
# Now create the umbrella sampler object 
#   


us = usample.UmbrellaSampler( log_prob_fn  , mpi=True, debug=True,  burn_acor=20 )

#
# Now add some umbrellas.
# First, define some temperatures to run with. 
#

temps = np.linspace( 1 , 10 , 8 ) 
centers = np.linspace( 0 , 1 , 8 )

#
# Then add an umbrella at each temperature. Use four walkers, and give some initial conditions
# Can be added individually, or in bulk:
#

#us.add_umbrellas( temperatures=temps , numwalkers=6 , ic=np.array([1,0]) , sampler=emcee.EnsembleSampler )
us.add_umbrellas( temperatures=temps, centers=centers, cvfn=cvfn , numwalkers=6 , ic=np.array([0,0]) , sampler=emcee.EnsembleSampler )

# 
# Then run for 10000 steps in each window.
# Output stats every [freq] steps
# Try to replica exchange [repex]-many walkers every [freq] steps
#

pos,weights,prob = us.run(500 , freq=500, repex=10    )

#
# We save the output
#

if (us.is_master() ):
    
    x = np.append( pos , weights , axis=1 )
    x = np.append( x , prob , axis=1 )

    np.savetxt(  'x.txt' , x )



us.close_pools()

