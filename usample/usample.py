import numpy as np
import random
import emus 
import mpi_pool
from mpi4py import MPI 

from umbrella import Umbrella
 
    

def SampleWindow(z): 
    
    ii,nsteps = z
    
    usampler.wlist[ii].sample(nsteps)
    
    return

def GetGR(ii):
    
    return usampler.wlist[ii].gr

def GetAcor(ii):
    
    #if (usampler.burn_acor>0):
    return usampler.wlist[ii].get_acor()
    #else:
    #    return 0

def GatherStates(ii):
    
    return usampler.wlist[ii].get_state()

def PushStates(z):
      
    ii,state = z
    
    usampler.wlist[ii].set_state(state)
    
    return  
 
def GatherTraj(ii):
    
    return usampler.wlist[ii].get_traj()

def PushTraj(z):
      
    ii,state = z
    
    usampler.wlist[ii].set_traj(state)
    
    return  

class UmbrellaSampler:
    
    def __init__(self, lpf, lpfargs=[], debug=False, evsolves=3, mpi=False, burn_pc=0.1, burn_acor=0):
        
        self.lpf = lpf
        self.lpfargs = lpfargs
         
        self.wlist = []
        
        self.debug = debug
        
        self.evsolves=evsolves
        
        self.mpi = mpi 
        self.us_pool = None
        
        self.staticpool = False
        
        self.burn_pc=burn_pc
        self.burn_acor=burn_acor
        
        self.zacor = []
        
        global usampler 
        usampler = self
         
        
    
    def add_umbrellas(self, temp_iterator, ic, numwalkers, sampler=None):
        
        
        self.w_comm = [None] * len( temp_iterator )
        self.wranks = [None] * len( temp_iterator )
             
        
        if (self.mpi):
            
            nproc = MPI.COMM_WORLD.Get_size()
            nwin = len( temp_iterator ) 
            
            if (nproc < nwin):
                self.us_comm = MPI.COMM_WORLD 
            else:
                self.group = MPI.COMM_WORLD.Get_group()
                us_group = self.group.Incl(  np.arange(0, nwin ) )
                self.us_comm = MPI.COMM_WORLD.Create( us_group )   
                
                self.wranks = [ range(ii,nproc,nwin) for ii in range(nwin) ]
                
                self.w_comm = []
                
                for s in self.wranks:
                    if len(s)>1:
                        sample_group = self.group.Incl( s )
                        self.w_comm.append(  MPI.COMM_WORLD.Create( sample_group )  )
                    else:
                        self.w_comm.append( None ) 
                     
        
        for ii, tt in enumerate(temp_iterator):
            
            self.add_umbrella(tt,ic,numwalkers,sampler, comm=self.w_comm[ii], ranks=self.wranks[ii] )
             
        if (self.debug and self.mpi):
            if (self.is_master() ):
                print "    [d]: Cores distributed as " + str( self.wranks )
        
             
        if (self.debug):
            if (self.is_master() ):
                print "    [d]: Temperatures: " + str( ["%.2f" % elem for elem in temp_iterator] )
            
    
    def add_umbrella(self , temp, ic , numwalkers , sampler=None, comm=None, ranks=None ):
        
        nu = Umbrella( self.lpf , temp , ic , numwalkers, lpfargs=self.lpfargs, sampler=sampler, comm=comm, ranks=ranks )
        
        self.wlist.append( nu )
          
            
    def run_repex(self, nrx):
        
        if (nrx<1):
            return
        
        
        if (self.us_pool):
            states = self.us_pool.map( GatherStates , range(0,len(self.wlist))  )
        else:
            states = map( GatherStates , range(0,len(self.wlist))  )
        
        for ii, z in enumerate( states ):
            self.wlist[ii].set_state(z)
        
        
        
        svec = np.zeros( len(self.wlist) -1 )
        
        evodd = np.arange( len( self.wlist)  - 1 )
        evodd = np.concatenate( (evodd[0::2] , evodd[1::2]) ) # 0 2 4 6 8 1 3 5 7
      
        for _ in np.arange( nrx ):
  
            for wn in evodd: 
            
                ii = random.randint( 0 , self.wlist[wn].nows  -1 )
                jj = random.randint( 0 , self.wlist[wn+1].nows  -1 )
                
                bias_i_in_i = self.wlist[wn].blobs0[ii] 
                bias_j_in_j = self.wlist[wn+1].blobs0[jj] 
                
                bias_i_in_j = self.wlist[wn+1].getbias( self.wlist[wn].lnprob0[ii] )
                bias_j_in_i = self.wlist[wn].getbias( self.wlist[wn+1].lnprob0[jj] )
                
                newE = bias_i_in_j + bias_j_in_i
                oldE = bias_i_in_i + bias_j_in_j
                
                logR = np.log(  random.random() )
                                
                if (logR < (newE - oldE) ):
                    
                    # Perform swap
                    
                    pos_i = np.copy( self.wlist[wn].p[ii] )
                    pos_j = np.copy( self.wlist[wn+1].p[jj] )
                    prob_i = self.wlist[wn].lnprob0[ii]
                    prob_j = self.wlist[wn+1].lnprob0[jj]
                    
                    self.wlist[wn].p[ii] = pos_j
                    self.wlist[wn].lnprob0[ii] = prob_j - bias_j_in_j + bias_j_in_i
                    self.wlist[wn].blobs0[ii] = bias_j_in_i
                    
                    self.wlist[wn+1].p[jj] = pos_i
                    self.wlist[wn+1].lnprob0[jj] = prob_i - bias_i_in_i + bias_i_in_j
                    self.wlist[wn+1].blobs0[jj] = bias_i_in_j
                    
                    svec[wn]+=1
                    
        states=[]
        
        for w in self.wlist:
            states.append( w.get_state() )
        
        
        if (self.us_pool):
            states = self.us_pool.map( PushStates , zip( range(0,len(self.wlist)) , states )  )
        else:
            states = map( PushStates , zip( range(0,len(self.wlist)) , states )  )
            
        if (self.debug):
            print "    [d]: Repex summary " + str( svec )
            
    
    def get_gr(self):
        
        if (self.us_pool):
            gr = self.us_pool.map( GetGR , range(0,len(self.wlist))  )
        else:
            gr = map( GetGR , range(0,len(self.wlist))  )
            
        return np.max( gr )
    
    def get_static_pool(self):
        
        if (not self.mpi):
            return None
        
        if (MPI.COMM_WORLD.Get_rank()>=len(self.wlist) ):
            return None
        
        
        self.staticpool = True
        
        self.us_pool = mpi_pool.MPIPool(comm=self.us_comm) 
        
        return self.us_pool
        
    
    def run(self, tsteps , freq=0, repex=0, grstop=0, OutputWeights=True ):
        
        steps = 0
        currentgr = 1.0
        
        if (freq<1):
            freq = tsteps
         
            
        if (self.mpi):
            
            if (MPI.COMM_WORLD.Get_rank()>=len(self.wlist) ):
                return (None,None,None)
            
            if (not self.staticpool):
                self.us_pool = mpi_pool.MPIPool(comm=self.us_comm) 
                
            if (MPI.COMM_WORLD.Get_rank()>0):
                if (self.staticpool):
                    return
                else:
                    self.us_pool.wait()
                    return (None,None,None)
                
                
        while (steps < tsteps) and ( currentgr > grstop  ):
            
            if (self.us_pool):
                self.us_pool.map( SampleWindow , zip( range(0,len(self.wlist)) , [freq]*len(self.wlist) ) )
            else:
                map( SampleWindow , zip( range(0,len(self.wlist)) , [freq]*len(self.wlist) ) )
            
            
            steps += freq
            if (self.debug and OutputWeights):
                print " :- Completed " + str(steps) + " of " + str(tsteps) + " iterations."
              
            currentgr = np.max( self.get_gr() ) 
            if (self.debug):
                print "    [d]: max gr: " + str( currentgr )
            
            self.run_repex( repex )
            
            
        
        if (self.us_pool):
            zacor = self.us_pool.map( GetAcor ,  range(0,len(self.wlist))   )
        else:
            zacor = map( GetAcor ,  range(0,len(self.wlist))   )
        
        self.zacor = zacor

        if (not OutputWeights):
            if (self.us_pool):
                if (not self.staticpool):
                    self.us_pool.close()
            return (None,None,None)


        if (self.us_pool):
            Traj = self.us_pool.map( GatherTraj ,  range(0,len(self.wlist))   )
            map( PushTraj , zip( range(0,len(self.wlist)) , Traj ) )
            if (not self.staticpool):
                self.us_pool.close()
            
            

        self.Solve_EMUS( self.evsolves )
        
        if (self.debug):
            print "    [d]: z values " + str( self.z )
            
        
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
            print "    [d]: Acor values " + str( ["%.2f" % elem for elem in self.zacor] )
            print "    [d]: Percentage burned: " + str(100 - 100.0 * np.sum(burnmask) / np.size(burnmask) ) + "%, (" + str( int(np.sum(1-burnmask) )) + " of " + str(np.size(burnmask)) + ")"
        
        wgt = np.zeros(  (len(prob) , 0 ) )
        widx = 0
        # <f> = < f  /  (\sum \psi_i / z_i ) >
        
        for w in self.wlist:
        
            wgt = np.append(wgt ,  w.getbias( prob ) - np.log( self.z[widx] ), axis=1)
            widx+=1
        
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
        
            
            
    def get_AvgPsi(self):
        
        NW = len(self.wlist)
        NS = np.size(self.wlist[0].traj_prob)
        Nwalkers = np.shape(self.wlist[0].traj_prob)[1] 
        
        AvgPsi = np.zeros( (NW , NS , NW ) )
        
        for w1 in range(NW):
        
            traj = np.array(self.wlist[w1].traj_prob).flatten()
            
            for  w2  in range(NW):
                
                AvgPsi[w1,:,w2] = self.wlist[w2].getbias( traj )
                
        AvgPsi = AvgPsi - np.max( AvgPsi )
        
        AP = []
        starts = []
        
        for ii in range(NW):
             
            
            st = int( self.burn_acor * self.zacor[ii]  * Nwalkers )
            
            
            if (st>=int(self.burn_pc*NS)):
                st = int(self.burn_pc*NS)
            if (st<0):
                st=0
                
            
            zz = AvgPsi[ii,(st):,:]
            zz = zz.squeeze()
            
            AP.append( np.exp( zz ) )
            
            starts.append( st )
        
        
        self.AvgPsi = np.exp( AvgPsi )
        
        self.AvgPsi = AP
        starts = np.array(starts) / Nwalkers
        
        self.starts = [ int(x) for x in starts ]
        
        
        
        return self.AvgPsi
    
    
    
    def Solve_EMUS(self, evsolves=3 ):
        
        AvgPsi = self.get_AvgPsi()
        
        self.z , self.F = emus.calculate_zs( AvgPsi, nMBAR=evsolves )
        
        return self.z, self.F
    
                
    def close_pools(self):
        
        if (not self.mpi):
            return
        
        if (self.us_pool):
            if (not self.staticpool):
                self.us_pool.close()
            
        for w in self.wlist:
            if (w.pool):
                w.pool.close()
        
    def is_master(self):
        if (not self.mpi):
            return True
        
        return MPI.COMM_WORLD.Get_rank()==0
     
                
                