#
# Samples a 2d Gaussian using Umbrella sampling in the temperature (serial version)
#
# Usage:
# > python gaussian.py
#


import usample.usample
import numpy as np
import emcee 

#
# Sample a 2D Gaussian
#
# Define the log probability function:
#


def log_prob_fn(p , means , icov ):
    
    pp = p - means
    
    lpf = - 0.5 * np.dot(pp , np.dot( icov , pp ) )
    
    return lpf

#
# Now create the umbrella sampler object 
#  

means = np.array([0.5,-0.25])

icov =  np.array([ [1,0.5] , [0.5,1] ])


us = usample.UmbrellaSampler( log_prob_fn , lpfargs=[means,icov],   debug=True,  burn_acor=20 )

#
# Now add some umbrellas.
# First, define some temperatures to run with. 
#

temps = np.linspace( 1 , 10 , 8 ) 

#
# Then add an umbrella at each temperature. Use four walkers, and give some initial conditions
# Can be added individually, or in bulk:
#

us.add_umbrellas( temperatures=temps , numwalkers=8 , ic=means , sampler=emcee.EnsembleSampler )

# 
# Then run for 10000 steps in each window.
# Output stats every [freq] steps
# Try to replica exchange [repex]-many walkers every [freq] steps
#

pos,weights,prob = us.run(10000 , freq=1000, repex=10   )

#
# We save the output
#
 
x = np.append( pos , weights , axis=1 )
x = np.append( x , prob , axis=1 )

np.savetxt(  'results_gaussian.txt' , x )
 

