"""
Umbrella sampling wrapper, making heavy use of the original EMUS code.
"""
import numpy as np
import random
try:
    import usample.emus as emus
    from .umbrella import Umbrella
    from .makecv import get_ic
except ImportError:
    import emus
    from umbrella import Umbrella
    from makecv import get_ic

def sample_window(z):     
    """
    Global helper function used to sample a given window.

    Parameters
    ----------
    z : list
        List indicating which umbrella to sample, z=[WindowIndex,NumberOfSteps,NumberToThin]. 
    """ 
    ii,nsteps,thin = z
    
    usampler.wlist[ii].sample(nsteps,thin=thin)
    
    return

def get_gr(ii):    
    """
    Return the GR of a given window.

    Parameters
    ----------
    ii : integer
        The index of the umbrella to compute the GR for.

    Returns
    -------
    gr : float
        The GR of window ii.
    """
    
    return usampler.wlist[ii].gr

def get_acor(ii):    
    """
    Return the Acor value of a given window.

    Parameters
    ----------
    ii : integer
        The index of the umbrella to compute the Acor value of.

    Returns
    -------
    acor : float
        The autocorrelation estimate for window ii.
    """
    
    return usampler.wlist[ii].get_acor()


def gather_states(ii):    
    """
    Return the current state (position/probability/bias) of walkers in a given window.

    Parameters
    ----------
    ii : integer
        The index of the umbrella to get the state of.

    Returns
    -------
    state : list
        The state of window ii.
    """
    
    return usampler.wlist[ii].get_state()

def push_states(z):    
    """
    Set the state of a given window.

    Parameters
    ----------
    z : list
        With z=[WindowIndex, State], this sets the state of umbrella WindowIndex to State. 
    """
      
    ii,state = z
    
    usampler.wlist[ii].set_state(state)
    
    return  
 
def gather_traj(ii):    
    """
    Return the current trajectory (position/probability/bias) of walkers in a given window.

    Parameters
    ----------
    ii : integer
        The index of the umbrella to get the trajectory of.

    Returns
    -------
    state : list
        The trajectory of window ii.
    """ 
    return usampler.wlist[ii].get_traj()

def push_traj(z):
    """
    Set the trajectory of a given window.

    Parameters
    ----------
    z : list
        With z=[WindowIndex, State], this sets the trajectory of umbrella WindowIndex to State. 
    """ 
    ii,state = z
    
    usampler.wlist[ii].set_traj(state)
    
    return  

class UmbrellaSampler:
    """
    A class container with helper functions for using the umbrella sampling method.
    """ 
    def __init__(self, lpf, lpfargs=[],lpfkwargs={} , debug=False, evsolves=3, mpi=False, burn_pc=0.1, burn_acor=0, logpsicutoff=700):
        """
        Initializer for the umbrella sampler class.

        Parameters
        ----------
        lpf : function
            The function handle for the log probability to be sampled.
        lpfargs : list (optional)
            Arguments for the LPF function.
        lpfkwargs : dictionary (optional)
            KwArgs for the LPF function.
        debug : boolean (optional)
            Whether to print useful debug information.
        evsolves : integer (optional)
            The number of eigenvector solve iterations to do in the EMUS scheme.
        mpi : boolean (optional)
            Whether to use MPI4PY to parallelize the sampling, or not.
        burn_pc : float (optional)
            The maximum fraction of the trajectory to be burn in initialization (0.1 corresponds to 10percent).
        burn_acor : integer (optional)
            The maximum number of 'acor'-times to be used for burn in.
        logpsicutoff : float (optional)
            The maximum value of the log of the biasing function that should be used as a cutoff.

        Returns
        -------
        usample : UmbrellaSampler
            The initialized umbrella sampler object.
        """        
        self.lpf = lpf
        self.lpfargs = lpfargs
        self.lpfkwargs = lpfkwargs
         
        self.wlist = []
        
        self.debug = debug
        
        self.evsolves=evsolves
        
        self.mpi = mpi 
        self.us_pool = None
        
        if (mpi):
            try:
                import usample.mpi_pool as mpi_pool
            except ImportError:
                import mpi_pool
            from mpi4py import MPI 
            self.MPI = MPI 
            self.mpi_pool = mpi_pool

        
        self.staticpool = False
        
        self.burn_pc=burn_pc
        self.burn_acor=burn_acor
        
        self.zacor = []
        self.z = 0
        
        self.logpsicutoff = logpsicutoff
        
        global usampler 
        usampler = self
        
        
    
    def add_umbrellas(self, temperatures=[1.0], centers=[0.0], cvfn=None, ic=None, numwalkers=None, sampler=None, samplerargs={}):
        """
        Add, and initialize (but don't sample from) some umbrella window distributions.
        Takes a list of temperatures and centers for stratifying in temperature and cv-space.

        Parameters
        ----------
        temperatures : list 
            A list of floats to be used as temperatures in the distributions.
        centers : list 
            A list of centers for the biasing distributions.
        cvfn : list
            A list specifying how the collective variable is defined.
        ic : Array 
            The initial point to sample from.
        numwalkers : integer
            The number of walkers to use in the umbrellas.
        sampler : ParallelSampler
            The sampler object to call to sample within each umbrella.
        samplerargs : dictionary
            Any additional arguments to pass to the sampler at sample-time. 
        """ 
        centers = np.array(centers)
        
        ntemps = len(temperatures)
        ncenters = len(centers)
            
        nwin = ntemps * ncenters
        
        self.w_comm = [None] * nwin
        self.wranks = [None] * nwin
             
        
        if (self.mpi): 
            
            nproc = self.MPI.COMM_WORLD.Get_size() 
            
            if (nproc < nwin):
                self.us_comm = self.MPI.COMM_WORLD 
            else:
                self.group = self.MPI.COMM_WORLD.Get_group()
                us_group = self.group.Incl(  np.arange(0, nwin ) )
                self.us_comm = self.MPI.COMM_WORLD.Create( us_group )   
                
                self.wranks = [ range(ii,nproc,nwin) for ii in range(nwin) ]
                
                self.w_comm = []
                
                for s in self.wranks:
                    if len(s)>1:
                        sample_group = self.group.Incl( s )
                        self.w_comm.append(  self.MPI.COMM_WORLD.Create( sample_group )  )
                    else:
                        self.w_comm.append( None ) 
                     
        ii=0
        sigmas=[]
        
        for jj,cc in enumerate(centers):
            
            if (cvfn is None or len(centers)<2):
                sigma = 1.0
                ic_cv = ic
            else:
                
                cvtype = cvfn[0]
                
                if (cvtype=="line"):
                    mdist = centers - cc
                    mdist = np.abs(mdist) 
                    sigma = np.min(mdist[mdist>0]) 
                
                if (cvtype=="grid"):
                    mdist = centers - cc
                    mdist2 = np.abs(mdist[:,1])
                    mdist1 = np.abs(mdist[:,0])
                    sigma  = np.array([np.min(mdist1[mdist1>0]), np.min(mdist2[mdist2>0]) ] )
                    
                    
                sigmas.append(sigma)
                ic_cv = get_ic( cc , cvfn )
                 
                
            for tt in temperatures:
                
                self.add_umbrella(ic_cv,numwalkers,sampler, comm=self.w_comm[ii], ranks=self.wranks[ii],center=cc,cvfn=cvfn,sigma=sigma, temp=tt, samplerargs=samplerargs )
                
                ii+=1
                

        
             
        if (self.debug):
            if (self.is_master() ):
                print("    [d]: Total windows: %s"%(str( nwin )))
                if (len(temperatures)>1):
                    print("    [d]: Temperatures: %s"%(str( temperatures )))
                if (len(centers)>1):
                    print("    [d]: Centers: %s"%(str( centers ))) 
                    print("    [d]: Sigmas: %s"%(str( np.array(sigmas))))
                    
                if (self.mpi):
                    print("    [d]: Cores distributed as %s"%(str( self.wranks )))
                     
    
    def add_umbrella(self , ic , numwalkers , sampler=None, comm=None, ranks=None, temp=None, center=None, cvfn=None,sigma=0, samplerargs={} ):
        """
        Adds one umbrella to the umbrella sampling list.

        Parameters
        ----------
        ic : Array 
            The initial point to sample from.
        numwalkers : integer
            The number of walkers to use in the umbrellas.
        sampler : ParallelSampler
            The sampler object to call to sample within each umbrella.
        comm : MPI4PY Communicator
            The Communicator object for this umbrella to use.
        ranks : list
            A list of integers specifying which MPI ranks should be used for this umbrella. 
        temp : float 
            The temperature that should be used for this distribution.
        center : float
            The central point of the umbrellas biasing distribution.
        cvfn : list
            A list specifying how the collective variable is defined.
        sigma : float
            The scale parameter for the biasing distribution.
        samplerargs : dictionary
            Any additional arguments to pass to the sampler at sample-time. 
        """ 
        nu = Umbrella( self.lpf , ic , numwalkers, lpfargs=self.lpfargs, lpfkwargs=self.lpfkwargs, sampler=sampler, comm=comm, ranks=ranks, temp=temp, center=center, cvfn=cvfn,sigma=sigma, samplerargs=samplerargs )
        
        self.wlist.append( nu )
          
    def get_z(self):
        """
        Returns the list of weights for the different windows.

        Returns
        -------
        z : list
            A list of floats corresponding to the relative weight of each window.
        """        
        return self.z
    
    def get_f(self):
        """
            Returns the F weighting matrix.
            
            Returns
            -------
            F : array
            The array of weights used in the eigenvector method for Umbrella Sampling.
            """
        return self.F
          
            
    def run_repex(self, nrx):
        """
        Performs a replica exchange (repex) step on the current set of umbrellas.

        Parameters
        ----------
        nrx : integer
            The number of replica exchange swaps to perform. 
        """   
        if (nrx<1):
            return
        
        
        if (self.us_pool):
            states = self.us_pool.map( gather_states , range(0,len(self.wlist))  )
        else:
            states = map( gather_states , range(0,len(self.wlist))  )
        
        for ii, z in enumerate( states ):
            self.wlist[ii].set_state(z)
        
        
        
        svec = np.zeros( len(self.wlist) )
        
        evodd = np.arange( len( self.wlist)  - 1 )
        evodd = np.concatenate( (evodd[0::2] , evodd[1::2]) ) # 0 2 4 6 8 1 3 5 7
      
        evoddplus1 = evodd+1
        
        
        attempts = 0
        accepts = 0
        
        for _ in np.arange( nrx ):
  
            for wi,wn in enumerate(evodd): 
            
                wnp1 = evoddplus1[ wi ]
            
                ii = random.randint( 0 , self.wlist[wn].nows  -1 )
                jj = random.randint( 0 , self.wlist[wnp1].nows  -1 )
                
                bias_i_in_i = self.wlist[wn].blobs0[ii] 
                bias_j_in_j = self.wlist[wnp1].blobs0[jj] 
                
                pi = self.wlist[wn].lnprob0[ii] - bias_i_in_i
                pj = self.wlist[wnp1].lnprob0[jj] - bias_j_in_j
                
                bias_i_in_j = self.wlist[wnp1].get_bias(  self.wlist[wn].p[ii] , pi )
                bias_j_in_i = self.wlist[wn].get_bias( self.wlist[wnp1].p[jj]  , pj )
                
                newE = bias_i_in_j + bias_j_in_i
                oldE = bias_i_in_i + bias_j_in_j
                
                logR = np.log(  random.random() )
                                
                if (logR < (newE - oldE) ):
                    
                    # Perform swap
                    
                    pos_i = np.copy( self.wlist[wn].p[ii] )
                    pos_j = np.copy( self.wlist[wnp1].p[jj] )
                    prob_i = self.wlist[wn].lnprob0[ii]
                    prob_j = self.wlist[wnp1].lnprob0[jj]
                    
                    self.wlist[wn].p[ii] = pos_j
                    self.wlist[wn].lnprob0[ii] = prob_j - bias_j_in_j + bias_j_in_i
                    self.wlist[wn].blobs0[ii] = bias_j_in_i
                    
                    self.wlist[wnp1].p[jj] = pos_i
                    self.wlist[wnp1].lnprob0[jj] = prob_i - bias_i_in_i + bias_i_in_j
                    self.wlist[wnp1].blobs0[jj] = bias_i_in_j
                    
                    svec[wn]+=1
                    accepts+= 1
                    
                attempts +=1
                    
        states=[]
        
        for w in self.wlist:
            states.append( w.get_state() )
        
        
        if (self.us_pool):
            states = self.us_pool.map( push_states , list(zip( range(0,len(self.wlist)) , states ))  )
        else:
            states = map( push_states , list(zip( range(0,len(self.wlist)) , states ))  )
            
        if (self.debug):
            accrate = accepts / (1.0*attempts)
            print("    [d]: Repex summary (%s %%):  %s"%(str(int(10000*accrate )/100.0), str( svec )))
            
    
    def get_gr(self):
        """
        Returns the maximum GR value from all umbrella windows.

        Returns
        -------
        gr : float
            The maximum GR statistic from all windows.
        """   
        if (self.us_pool):
            gr = self.us_pool.map( get_gr , range(0,len(self.wlist))  )
        else:
            gr = map( get_gr , range(0,len(self.wlist))  )
            
        return np.max( gr )
    
    def get_static_pool(self):
        """
        Allows access to the static MPIPool used by the sampler.
 
        Returns
        -------
        uspool : MPIPool
            The MPIPool object used by the umbrella sampler.
        """  
        if (not self.mpi):
            return None
        
        if (self.MPI.COMM_WORLD.Get_rank()>=len(self.wlist) ):
            return None
        
        
        self.staticpool = True
        
        self.us_pool = self.mpi_pool.MPIPool(comm=self.us_comm) 
        
        return self.us_pool
        
    
    def run(self, tsteps , freq=0, repex=0, grstop=0, output_weights=True, thin=1 ):
        """
        Rub the sampling on the umbrellas.

        Parameters
        ----------
        tsteps : integer
            The number of steps to run, per walker, per window. 
        freq : integer
            The sampler will run bursts of 'freq' steps, calculating the GR and other statistics in between.
        repex : integer
            The number of attempted replica exchange steps every 'freq' steps.
        grstop : float
            The sampler will stop if the maximum GR falls below this value.
        output_weights : boolean
            Whether to compute the umbrella weights at the end of sampling. 
            One only needs to compute the weights if all sampling is completed.
        thin : integer
            The number of steps to thin the trajectory by, eg thin=3 only uses every third step.
            

        Returns
        -------
        pos : list
            If output_weights is True, returns a list of trajectories for each walker over all windows.
        wgt : list
            If output_weights is True, returns a list of sample weights for each point.
        prob : list
            If output_weights is True, returns a list of the log probabilities of each sampled point.
        """  
        steps = 0
        currentgr = 1.0
        
        if (freq<1):
            freq = tsteps
         
            
        if (self.mpi):
            
            if (self.MPI.COMM_WORLD.Get_rank()>=len(self.wlist) ):
                return (None,None,None)
            
            if (not self.staticpool):
                self.us_pool = self.mpi_pool.MPIPool(comm=self.us_comm) 
                
            if (self.MPI.COMM_WORLD.Get_rank()>0):
                if (self.staticpool):
                    return
                else:
                    self.us_pool.wait()
                    return (None,None,None)
                
                
        while (steps < tsteps) and ( currentgr > grstop  ):
            if (self.us_pool):
                self.us_pool.map( sample_window , list(zip( range(0,len(self.wlist)) , [freq]*len(self.wlist), [thin]*len(self.wlist) )) )
            else:
                map( sample_window , list(zip( range(0,len(self.wlist)) , [freq]*len(self.wlist), [thin]*len(self.wlist) )) )
            
            
            steps += freq
            if (self.debug and output_weights):
                print(" :- Completed %s of %s iterations."%(str(steps), str(tsteps)))
              
            currentgr = np.max( self.get_gr() ) 
            if (self.debug):
                print("    [d]: max gr: %s"%str( currentgr )
)            
            self.run_repex( repex )
            
            
        
        if (self.us_pool):
            zacor = self.us_pool.map( get_acor ,  range(0,len(self.wlist))   )
        else:
            zacor = map( get_acor ,  range(0,len(self.wlist))   )
        
        self.zacor = zacor

        if (not output_weights):
            if (self.us_pool):
                if (not self.staticpool):
                    self.us_pool.close()
            return (None,None,None)


        if (self.us_pool):
            Traj = self.us_pool.map( gather_traj ,  range(0,len(self.wlist))   )
            map( push_traj , list(zip( range(0,len(self.wlist)) , Traj )) )
            if (not self.staticpool):
                self.us_pool.close()
            
            

        self.solve_emus( self.evsolves )
        
        if (self.debug):
            print("    [d]: z values %s"%str( self.z ))
            
        
        pos = np.zeros( (np.shape( self.wlist[0].traj_pos)[0] , 0 , np.shape( self.wlist[0].traj_pos)[2] ) )
        prob = np.zeros( (np.shape( self.wlist[0].traj_pos)[0] , 0  ) )
        burnmask = np.zeros( (np.shape( self.wlist[0].traj_pos)[0] , 0  ) ) 
        
        for ii,w in enumerate(self.wlist):
            
            traj = np.array( w.traj_pos )
            
            pos = np.append( pos , traj , axis=1 )
            
           
            
            traj_prob =  np.array(w.traj_prob).squeeze()
            
            bb = np.ones(  np.shape( traj_prob ) )
             
            bb[:self.starts[ii],:] = 0 
             
            
            prob = np.append( prob , traj_prob , axis=1 )
            
            burnmask = np.append( burnmask , np.copy( bb ) , axis=1 )
                
        
        ts=np.shape(pos)
        
        pos = np.reshape( pos , (ts[0]*ts[1],ts[2] ) )
         
        
        prob = np.reshape( prob , (ts[0]*ts[1],1) )
        
        burnmask = np.reshape( burnmask , (ts[0]*ts[1],1) )
        
        if (self.debug  ):
            print("    [d]: Acor values %s"%(str( [float("%.2f" % elem) for elem in self.zacor] )))
            print("    [d]: Percentage burned: %s %% %s of %s )"%(str(100 - 100.0 * np.sum(burnmask) / np.size(burnmask) ), str( int(np.sum(1-burnmask) )), str(np.size(burnmask))))
        
        wgt = np.zeros(  (len(prob) , 0 ) )
        widx = 0
        # <f> = < f  /  (\sum \psi_i / z_i ) >
        
        #cutvals = np.zeros( len( self.wlist ) )
        
        for w in self.wlist:
         
            myres = w.get_bias( pos.T , prob )
            #cutvals[widx] = np.sum( myres.flatten()>self.logpsicutoff )
            #myres = np.fmin( myres , self.logpsicutoff )
            if (myres.ndim==1):
                myres = np.expand_dims( myres , axis=1 )
                
            wgt = np.append(wgt ,  myres - np.log( self.z[widx] ), axis=1)
            widx+=1
        
        #print("    [d]: Number cut= %s"%str( cutvals ))
        
        
        # Requires numpy 1.7+
        #maxW = np.max( wgt , axis=1, keepdims=True )
        
        maxW = np.max(wgt , axis=1)
        maxW = np.reshape( maxW , ( len(maxW),1) ) 
        
        wgt = wgt - maxW
        denom = np.sum(  np.exp( wgt ), axis = 1  ) 
        denom = np.reshape( denom , ( len(denom) , 1 ) )
        wgt = 1.0 / denom
        maxW = - maxW 
        maxW = maxW - np.max( maxW )
        wgt = wgt * np.exp( maxW ) 
         
        wgt = wgt * burnmask
         
        
        return  (pos, wgt, prob)
        
            
            
    def get_avg_psi(self):
        """
        Computes the average of the biasing functions psi. 
        
        Returns
        -------
        avgpsi : list
            Averages of each biasing function in each umbrella.
        """ 
        NW = len(self.wlist)
        NS = np.size(self.wlist[0].traj_prob)
        Nwalkers = np.shape(self.wlist[0].traj_prob)[1] 
        
        AvgPsi = np.zeros( (NW , NS , NW ) )
        
        for w1 in range(NW):
        
            traj_prob = np.array(self.wlist[w1].traj_prob).flatten()
            #traj_pos = np.array(self.wlist[w1].traj_pos).flatten()
            shp = np.shape( np.array(self.wlist[w1].traj_pos) )
            traj_pos = np.reshape( np.array(self.wlist[w1].traj_pos) , ( shp[0]*shp[1],shp[2] ) )
            traj_pos = traj_pos.T
             
            
            for  w2  in range(NW):
                
                AvgPsi[w1,:,w2] = self.wlist[w2].get_bias( traj_pos , traj_prob )
                

        self.maxpsi = np.zeros( NW ) 

#if (0): #for jj in range(NW):
 
 #zz = AvgPsi[:,:,jj].flatten()
 #kk = np.isfinite(zz)
 #self.maxpsi[jj] = np.min( zz[kk] )
 #AvgPsi[:,:,jj] = AvgPsi[:,:,jj] - self.maxpsi[jj]
                
        #AvgPsi = np.fmin( AvgPsi , self.logpsicutoff )
        
        AP = []
        starts = []
            #if (self.debug):
            #print("    [d]: winpsi=%s"%self.maxpsi)
        
        for ii in range(NW):
             
            
            st = int( self.burn_acor * self.zacor[ii]  * Nwalkers )
            
            
            if (st>=int(self.burn_pc*NS)):
                st = int(self.burn_pc*NS)
            if (st<0):
                st=0
                
            
            zz = AvgPsi[ii,(st):,:]
            zz = zz.squeeze()
            
            #AP.append( np.exp( zz ) )
            AP.append(  zz )
            
            starts.append( st )
        
        
        #self.AvgPsi = np.exp( AvgPsi )
        
        self.AvgPsi = AP
        starts = np.array(starts) / Nwalkers
        
        self.starts = [ int(x) for x in starts ]
        
        
        
        return self.AvgPsi
    
    
    
    def solve_emus(self, evsolves=3 ):
        """
        Computes the weights and overlap matrix used in the EMUS method.

        Parameters
        ----------
        evsolves : integer
            The number of eigenvalue iterations used in the EMUS method.

        Returns
        -------
        z : 1d array
            The relative weights for the umbrella windows.
        F : 2d array
            The overlap matrix, which has unit eigenvector z.
        """ 
        AvgPsi = self.get_avg_psi()
        
        self.z , self.F = emus.calculate_zs( AvgPsi, nMBAR=evsolves )
        
        return self.z, self.F
    
                
    def close_pools(self):
        """
        Closes the MPI pools to prevent dangling threads.
        """ 
        if (not self.mpi):
            return
        
        if (self.us_pool):
            if (not self.staticpool):
                self.us_pool.close()
            
        for w in self.wlist:
            if (w.pool):
                w.pool.close()
        
    def is_master(self):
        """
        Helper function to discern if a thread is the master thread or not.

        Returns
        -------
        ismaster : boolean
            Whether the thread is the master thread (==rank 0) or not.
        """ 
        if (not self.mpi):
            return True
        
        return self.MPI.COMM_WORLD.Get_rank()==0
     
                
                
