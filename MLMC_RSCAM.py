"""
MLMC_RSCAM
Multilevel Monte Carlo Implementation for RSCAM Group 1
Luke Shaw, Olena Balan, Isabell Linde, Chantal Kool, Josh Mckeown
_______________________________________________________
Functions: Euro_payoff, Asian_payoff, Lookback_payoff, Digital_payoff, and anti_Euro_payoff payoffs.
(diffusion/JD)_path(_min/_avg) for diffusion and jump diffusion, coarse/fine final, avg, min asset prices.
(diffusion/JD)_asset_plot plotting functions for diffusion and jump diffusion asset prices.
Giles_plot plotting functions for mlmc variance/mean, samples per level/complexity and brownian_plot for discretised Brownian motion plots.
Use inspect.getmembers(mlmc_RSCAM,inspect.isroutine) to get full list.


Classes: Option, JumpDiffusion_Option, Diffusion_Option, Merton_Option, GBM_Option, MyOption.
With specific Euro_GBM/Euro_Merton, Lookback_GBM/Lookback_Merton, Asian_GBM/Asian_Merton, 
Digital_GBM/Digital_Merton implementations for Merton and GBM models. 
Use inspect.getmembers(mlmc_RSCAM,inspect.isclass) to get full list.

________________________________________________________
Example usage:
import MLMC_RSCAM as multi

opt = multi.Euro_GBM(X0=125,K=105,r=0.02,sig=0.5)
sums,N=opt.mlmc(eps=0.01)
print(sum(sums[0,:]/N),opt.BS()) #Compare BS price with mlmc-calculated price

eps=[0.005,0.1,0.2,0.25,0.3]
fig,ax=plt.subplots(2,2,figsize=(30,30))
markers=['o','s','x','d']
multi.Giles_plot(opt,eps,label='European GBM ', markers=markers,fig=fig,Nsamples=10**3) #Plot mean/variance/numbr of levels/complexity plot
opt.asset_plot(L=4,M=4) #Plot asset price on two discretisation levels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as ticker
from scipy.special import factorial
from matplotlib.legend import Legend
import weakref

class WeakMethod:
    """
    Taken from:
    https://stackoverflow.com/questions/55413060/python-passing-functions-as-arguments-to-initialize-the-methods-of-an-object-p
    Need to use for passing functions with self as argument in constructor.
    """
    def __init__(self, func, instance):
        self.func = func
        self.instance_ref = weakref.ref(instance)

        self.__wrapped__ = func  # this makes things like `inspect.signature` work

    def __call__(self, *args, **kwargs):
        instance = self.instance_ref()
        return self.func(instance, *args, **kwargs)

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({!r}, {!r})'.format(cls_name, self.func, self.instance_ref())
#######################################################################################################################
##General Payoff Funcs
def EA_payoff(self,N_loop,l,M):
    """
    Payoff function for European Call or Asian Call option. Depends on what self.path returns.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """
    r=self.r
    K=self.K
    T=self.T
    Xf,Xc=self.path(N_loop,l,M)
    #Calculate payoffs etc.
    Pf=np.maximum(0,Xf-K)
    Pf=np.exp(-r*T)*Pf
    if l==0:
        return Pf,Xc #Just ignore Pc=Xc
    else:
        Pc=np.maximum(0,Xc-K)
        Pc=np.exp(-r*T)*Pc #Payoff at coarse level
        return Pf,Pc
        
        
def Lookback_payoff(self,N_loop,l,M):
    """
    Payoff function for Lookback Call option.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
         Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """
    r=self.r;T=self.T;beta=self.beta
    dt=T/M**l
    Xf,Mf,Xc,Mc=self.path(N_loop,l,M)
    #Calculate payoffs etc.
    Pf=Xf - Mf*(1-self.beta*self.sig*np.sqrt(dt))
    Pf=np.exp(-r*T)*Pf #Payoff at fine level
    if l==0:
        return Pf,Xc #Just ignore Pc
    else:
        Pc=Xc - Mc*(1-self.beta*self.sig*np.sqrt(M*dt))
        Pc=np.exp(-r*T)*Pc #Payoff at coarse level
        return Pf,Pc
        
def Digital_payoff(self,N_loop,l,M):
    """
    Payoff function for Digital Call option.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
         Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """
    r=self.r
    T=self.T
    K=self.K
    Xf,Xc=self.path(N_loop,l,M)
    #Calculate payoffs etc.
    Pf=np.exp(-r*T)*K*(Xf>K).astype(np.int_)
    if l==0:
        return Pf,Xc #Just ignore Pc=Xc
    else:
        Pc=np.exp(-r*T)*K*(Xc>K).astype(np.int_) #Payoff at coarse level
        return Pf,Pc

def Amer_payoff(self,N_loop,l,M):
    """
    Payoff function for American Put option. self.path should return Xf,Xc - N_loop-by-3 arrays.
    For a non-dividend-paying asset, an American Call will have the same value as a Euro Call.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=0 if l==0)
    """
    K=self.K;T=self.T;r=self.r
    Xf,Xc=self.path(N_loop,l,M)
    Pf = Xf[:,2] + Xf[:,1]*np.maximum(K-Xf[:,0],0)*np.exp(-r*T)
    if l == 0:
        return Pf,0 #ignore Pc
    else:
        Pc = Xc[:,2] + Xc[:,1]*np.maximum(K-Xc[:,0],0)*np.exp(-r*T)
        return Pf,Pc

#######################################################################################################################
#Antithetic funcs
def diffusion_anti_path(self,N_loop,l,M):
    """
    Antithetic path function for diffusion sde.
    In usage should be called by anti_payoff as self.anti_path. Calls self.sde.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Xa,Xc (numpy.array) : final asset price vectors for N_loop sample paths (Xc=X0 if l==0)
    """
    r=self.r;X0=self.X0;T=self.T;K=self.K;sig=self.sig
    sde=self.sde
    
    if M%2!=0:
        raise Exception("Cannot calculate antithetic estimator for odd coarseness factor M - please use even M")
        
    Nsteps = M**l #Number of fine steps
    dt = T / Nsteps
    sqrt_dt = np.sqrt(dt)
    
    # Initialise fine, coarse, antithetic asset prices; 
    Xf = X0 * np.ones(N_loop)
    Xc = X0 * np.ones(N_loop)
    Xa = X0 * np.ones(N_loop)

    # coarse Brownian increment (BI)
    dWc = np.zeros(N_loop)

    # this loop generates brownian increments and calculates the estimators
    for j in range(2, Nsteps+1, 2): #If l==0, Nsteps=1 and loop is skipped
        dWf_odd = np.random.randn(N_loop) * sqrt_dt
        dWf_even = np.random.randn(N_loop) * sqrt_dt

        # odd step
        t_=j*dt
        Xf += sde(Xf,dt,t_,dWf_odd)
        Xa += sde(Xa,dt,t_,dWf_even)
        
        t_+=dt
        # even step
        Xf += sde(Xf,dt,t_,dWf_even)
        Xa += sde(Xa,dt,t_,dWf_odd)

        dWc += dWf_odd + dWf_even
        if j%M==0:
            Xc += sde(Xc,M*dt,j*dt,dWc) #...Develop coarse path
            dWc = np.zeros(N_loop) #...Re-initialise coarse BI to 0

    if l == 0: #Loop has been skipped
        Xf += sde(Xf,dt,0,np.random.randn(N_loop)*sqrt_dt)
        return Xf,Xf,Xc #Just ignore Xa=Xf,Xc=X0
    else:
        return Xf,Xa,Xc
    
def diffusion_anti_path_avg(self,N_loop,l,M):
    """
    Antithetic path function for diffusion sde. Calculates antithetic path averages.
    In usage should be called by anti_payoff as self.anti_path. Calls self.sde.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
         Af/T,Aa/T,Ac/T (numpy.array) : final average asset price vectors for N_loop sample paths (Ac=X0 if l==0)
    """
    r=self.r;X0=self.X0;T=self.T;K=self.K;sig=self.sig
    sde=self.sde
    
    if M%2!=0:
        raise Exception("Cannot calculate antithetic estimator for odd coarseness factor M - please use even M")
        
    Nsteps = M**l #Number of fine steps
    dt = T / Nsteps
    sqrt_dt = np.sqrt(dt)
    
    # Initialise fine, coarse, antithetic asset prices; 
    Xf = X0 * np.ones(N_loop)
    Xc = X0 * np.ones(N_loop)
    Xa = X0 * np.ones(N_loop)
    
    Af=0.5*dt*Xf
    Aa=0.5*dt*Xa
    Ac=0.5*M*dt*Xc

    # coarse Brownian increment (BI)
    dWc = np.zeros(N_loop)

    # this loop generates brownian increments and calculates the estimators
    for j in range(2, Nsteps+1, 2): #If l==0, Nsteps=1 and loop is skipped
        dWf_odd = np.random.randn(N_loop) * sqrt_dt
        dWf_even = np.random.randn(N_loop) * sqrt_dt

        # odd step
        t_=j*dt
        Xf += sde(Xf,dt,t_,dWf_odd)
        Xa += sde(Xa,dt,t_,dWf_even)
        Af+=dt*Xf #fine average
        Aa+=dt*Xa #antithetic average
        
        t_+=dt
        
        # even step
        Xf += sde(Xf,dt,t_,dWf_even)
        Xa += sde(Xa,dt,t_,dWf_odd)
        Af+=dt*Xf #fine average
        Aa+=dt*Xa #antithetic average
        
        dWc += dWf_odd + dWf_even

        if j%M==0:
            Xc += sde(Xc,M*dt,j*dt,dWc) #...Develop coarse path
            Ac+=dt*M*Xc #coarse average
            dWc = np.zeros(N_loop) #...Re-initialise coarse BI to 0
            
    
    if l == 0: #Loop has been skipped
        Xf += sde(Xf,dt,0,np.random.randn(N_loop)*sqrt_dt)
        Af+=0.5*dt*Xf
        return Af/T,Af/T,Xc #Just ignore Xa=Xf,Xc=X0
    else: #Correct for not halving the final X value
        Af-=0.5*Xf*dt
        Aa-=0.5*Xa*dt
        Ac-=0.5*Xc*(M*dt)
        return Af/T,Aa/T,Ac/T
    
def anti_EA_payoff(self,N_loop,l,M):
    """
    Antithetic payoff function for European Call or Asian Call option. Depends on what self.path returns.
    
    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """
    r=self.r;K=self.K;T=self.T
    Xf,Xa,Xc=self.anti_path(N_loop,l,M) #call anti_path function (returns Xf,Xa,Xc if Euro, Af/T,Aa/T,Ac/T if Asian)
    
    #Calculate payoffs etc.
    Pf=np.maximum(0,Xf-K)
    Pf=np.exp(-r*T)*Pf #Payoff at fine level
    if l==0:
        return Pf,Xc #Just ignore Pc=Xc
    else:
        Pa=np.maximum(0,Xa-K)
        Pa=np.exp(-r*T)*Pa #Antithetic payoff
        Pc=np.maximum(0,Xc-K)
        Pc=np.exp(-r*T)*Pc #Payoff at coarse level
        return 0.5 * (Pf + Pa), Pc
#######################################################################################################################
##Path funcs
def diffusion_path(self,N_loop,l,M):
    """ 
    The path function for Euler-Maruyama diffusion, which calculates final asset prices X(T).

    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Xc (numpy.array) : final asset price vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """
    T=self.T;X0=self.X0;sde=self.sde
    Nsteps=M**l
    dt=T/Nsteps
    sqrt_dt=np.sqrt(dt)

    #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
    Xf=X0*np.ones(N_loop)
    Xc=X0*np.ones(N_loop)
    dWc=np.zeros(N_loop)
    for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
        t_=(j-1)*dt #Current time to simulate from in Ito calculus
        dWf=np.random.randn(N_loop)*sqrt_dt
        dWc=dWc+dWf #Keep adding to coarse BI every loop until j is integer multiple of M
        Xf+=sde(Xf,dt,t_,dWf)
        if j%M==0: #if j is integer multiple of M...
            Xc+=sde(Xc,M*dt,t_,dWc) #...Develop coarse path
            dWc=np.zeros(N_loop) #...Re-initialise coarse BI to 0
    return Xf,Xc

def diffusion_path_avg(self,N_loop,l,M):
    """ 
    The path function for Euler-Maruyama diffusion, which calculates final average asset price avg(X).

    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Af/T,Ac/T (numpy.array) : final average asset price vectors for N_loop sample paths (Ac=Xc=X0 if l==0)
    """
    T=self.T;X0=self.X0
    sde=self.sde
    Nsteps=M**l;dt=T/Nsteps;sqrt_dt=np.sqrt(dt)
    #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
    Xf=X0*np.ones(N_loop)
    Xc=X0*np.ones(N_loop)
    Af=0.5*dt*Xf
    Ac=0.5*M*dt*Xc
    dWc=np.zeros(N_loop)
    for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
        t_=(j-1)*dt #Current time to simulate from in Ito calculus
        dWf=np.random.randn(N_loop)*sqrt_dt
        dWc=dWc+dWf #Keep adding to coarse BI every loop until j is integer multiple of M
        Xf+=sde(Xf,dt,t_,dWf)
        Af+=Xf*dt
        if j%M==0: #if j is integer multiple of M...
            Xc+=sde(Xc,M*dt,t_,dWc) #...Develop coarse path
            Ac+=Xc*M*dt
            dWc=np.zeros(N_loop) #...Re-initialise coarse BI to 0

    Af-=0.5*Xf*dt
    Ac-=0.5*Xc*M*dt

    return Af/T,Ac/T

def diffusion_path_min(self,N_loop,l,M):
    """ 
    The path function for Lookback Call Option with Euler-Maruyama diffusion, which calculates final asset prices and
    minima.

    Parameters:
        self(Option): option that function is called through
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Mf,Xc,Mc (numpy.array) : final asset price vectors Xf,Xc and minima Mf,Mc for N_loop sample paths (Xc=Mc=X0 if
                                     l==0)
    """
    T=self.T;X0=self.X0
    sde=self.sde
    Nsteps=M**l
    dt=T/Nsteps
    sqrt_dt=np.sqrt(dt)

    #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
    Xf=X0*np.ones(N_loop)
    Xc=X0*np.ones(N_loop)
    Mf=X0*np.ones(N_loop)
    Mc=X0*np.ones(N_loop)
    dWc=np.zeros(N_loop)

    for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
        t_=(j-1)*dt #Current time to simulate from in Ito calculus
        dWf=np.random.randn(N_loop)*sqrt_dt
        dWc=dWc+dWf #Keep adding to coarse BI every loop until j is integer multiple of M
        Xf+=sde(Xf,dt,t_,dWf)
        Mf=np.minimum(Xf,Mf)
        if j%M==0: #if j is integer multiple of M...
            Xc+=sde(Xc,M*dt,t_,dWc) #...Develop coarse path
            Mc=np.minimum(Xc,Mc)
            dWc=np.zeros(N_loop) #...Re-initialise coarse BI to 0
            
    return Xf,Mf,Xc,Mc

def JD_path(self,N_loop,l,M):
    """
    The path function for Euler-Maruyama JD, which calculates final asset price X(T).

    Parameters:
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Xc (numpy.array) : final asset price vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
    """

    lam=self.lam;X0=self.X0;T=self.T;
    Xf=np.zeros(N_loop)
    Xc=np.zeros(N_loop)
    num=0
    Nsteps=M**l
    sde=self.sde
    jumpsize=self.jumpsize
    jumptime=self.jumptime

    while num<N_loop:
        #Initialise asset price and time
        dWc=0;tau=0;tf=0;tc=0;dtc=0
        Sf=X0;Sc=X0;
        ##Algorithm Start
        tau+=jumptime(scale=1/lam)
        for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
            tn=j*T/Nsteps #Fine timestepping right boundary
            while tau<tn: #If jump is before right boundary...
                dt=tau-tf #Adaptive step size is from last jump or left fine timestep
                dtc+=dt #Coarse timestep increments
                dWf=np.random.randn()*np.sqrt(dt) #Brownian motion for adaptive time step
                dWc+=dWf #Increment coarse BI
                dJ=np.exp(jumpsize())-1 #Generate jump

                #Develop fine path
                Sf+=sde(Sf,dt,tf,dWf,dJ)

                #Develop coarse path
                Sc+=sde(Sc,dtc,tc,dWc,dJ)

                dWc=0 #Reset coarse BI
                dtc=0 #Reset coarse timestep
                tf=tau #Both fine and coarse paths now at t_=latest jump time
                tc=tau
                tau+=jumptime(scale=1/lam) #Next jump time

            #Next jump time is after current right fine timestep
            dt=tn-tf #Adaptive time step is time from recent jump or left fine time up to right fine time
            dtc+=dt #Increment coarse timestep
            dWf=np.random.randn()*np.sqrt(dt) #Fine BI for adaptive timestep
            dWc+=dWf #Increment coarse BI
            Sf+=sde(Sf,dt,tf,dWf) #Develope fine timestep
            tf=tn #Fine path now at j*T/Nsteps, set as left boundary
            if j%M==0: #If reached coarse timepoint, then bring coarse path up to this point
                Sc+=sde(Sc,dtc,tc,dWc)#...Develop coarse path
                tc=tn #Coarse path now at j*T/Nsteps
                dtc=0
                dWc=0 #...Re-initialise coarse BI to 0

        Xf[num]=Sf
        Xc[num]=Sc
        num+=1 #One more simulation down
    return Xf,Xc

def JD_path_avg(self,N_loop,l,M):
    """
    The path function for Euler-Maruyama JD, which calculates average asset price over path, avg(X).

    Parameters:
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Af/T,Ac/T (numpy.array) : final average asset price vectors for N_loop sample paths (Ac=Xc=X0 if l==0)
    """
    lam=self.lam;X0=self.X0;T=self.T;
    Af=np.zeros(N_loop)
    Ac=np.zeros(N_loop)
    num=0
    Nsteps=M**l
    sde=self.sde
    jumpsize=self.jumpsize
    jumptime=self.jumptime
    
    while num<N_loop:

        #Initialise asset price and time
        dWc=0;tau=0;tf=0;tc=0;dtc=0
        Sf=X0;Sc=X0;
        avg_f=0;avg_c=0
        ##Algorithm Start
        tau+=jumptime(scale=1/lam)

        for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
            tn=j*T/Nsteps #Fine timestepping right boundary
            while tau<tn: #If jump is before right boundary...
                dt=tau-tf #Adaptive step size is from last jump or left fine timestep
                dtc+=dt #Coarse timestep increments
                dWf=np.random.randn()*np.sqrt(dt) #Brownian motion for adaptive time step
                dWc+=dWf #Increment coarse BI
                dJ=np.exp(jumpsize())-1 #Generate jump

                avg_f+=0.5*dt*Sf #S_n
                Sf+=sde(Sf,dt,tf,dWf) #Develop fine path
                avg_f+=0.5*dt*Sf #S-_n+1
                Sf+=Sf*dJ

                avg_c+=0.5*dtc*Sc #S_n
                Sc+=sde(Sc,dtc,tc,dWc) #Develop coarse path
                avg_c+=0.5*dtc*Sc #S-_n+1
                Sc+=Sc*dJ

                dWc=0 #Reset coarse BI
                dtc=0 #Reset coarse timestep
                tf=tau #Both fine and coarse paths now at t_=latest jump time
                tc=tau
                tau+=jumptime(scale=1/lam) #Next jump time

            #Next jump time is after current right fine timestep
            dt=tn-tf #Adaptive time step is time from recent jump or left fine time up to right fine time
            dtc+=dt #Increment coarse timestep
            dWf=np.random.randn()*np.sqrt(dt) #Fine BI for adaptive timestep
            dWc+=dWf #Increment coarse BI
            
            avg_f+=0.5*dt*Sf #S_n
            Sf+=sde(Sf,dt,tf,dWf) #Develope fine timestep
            avg_f+=0.5*dt*Sf #S-_n+1
            tf=tn #Fine path now at j*T/Nsteps, set as left boundary

            if j%M==0: #If reached coarse timepoint, then bring coarse path up to this point
                avg_c+=0.5*dtc*Sc #S_n
                Sc+=sde(Sc,dtc,tc,dWc)#...Develop coarse path
                avg_c+=0.5*dtc*Sc #S-_n+1
                tc=tn #Coarse path now at j*T/Nsteps
                dtc=0
                dWc=0 #...Re-initialise coarse BI to 0

        Af[num]=avg_f
        Ac[num]=avg_c
        num+=1 #One more simulation down

    return Af/T,Ac/T

def JD_path_min(self,N_loop,l,M):
    """
    The path function for Euler-Maruyama JD, which calculates final asset prices Xf,Xc and minima Mf,Mc.

    Parameters:
        N_loop(int): total number of sample paths to evaluate payoff on
        l(int) : discretisation level
        M(int) : coarseness factor, number of fine steps = M**l
    Returns:
        Xf,Mf,Xc,Mc (numpy.array) : final asset price vectors Xf,Xc and minima Mf,Mc for N_loop sample paths (Xc=Mc=X0 if l==0)
    """
    lam=self.lam;X0=self.X0;T=self.T;
    Xf=np.zeros(N_loop)
    Xc=np.zeros(N_loop)
    Mf=np.zeros(N_loop)
    Mc=np.zeros(N_loop)
    num=0
    Nsteps=M**l
    sde=self.sde
    jumpsize=self.jumpsize
    jumptime=self.jumptime

    while num<N_loop:

        #Initialise asset price and time
        dWc=0;tau=0;tf=0;tc=0;dtc=0
        Sf=X0;Sc=X0;
        ##Algorithm Start
        tau+=jumptime(scale=1/lam)
        mf=X0
        mc=X0

        for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
            tn=j*T/Nsteps #Fine timestepping right boundary
            while tau<tn: #If jump is before right boundary...
                dt=tau-tf #Adaptive step size is from last jump or left fine timestep
                dtc+=dt #Coarse timestep increments
                dWf=np.random.randn()*np.sqrt(dt) #Brownian motion for adaptive time step
                dWc+=dWf #Increment coarse BI
                dJ=np.exp(jumpsize())-1 #Generate jump

                #Develop fine path
                Sf+=sde(Sf,dt,tf,dWf,dJ)

                #Develop coarse path
                Sc+=sde(Sc,dtc,tc,dWc,dJ)

                #Update minima
                mf=min(mf,Sf)
                mc=min(mc,Sc)

                dWc=0 #Reset coarse BI
                dtc=0 #Reset coarse timestep
                tf=tau #Both fine and coarse paths now at t_=latest jump time
                tc=tau
                tau+=jumptime(scale=1/lam) #Next jump time

            #Next jump time is after current right fine timestep
            dt=tn-tf #Adaptive time step is time from recent jump or left fine time up to right fine time
            dtc+=dt #Increment coarse timestep
            dWf=np.random.randn()*np.sqrt(dt) #Fine BI for adaptive timestep
            dWc+=dWf #Increment coarse BI
            Sf+=sde(Sf,dt,tf,dWf) #Develope fine timestep
            tf=tn #Fine path now at j*T/Nsteps, set as left boundary
            mf=min(mf,Sf) #Update minimum
            if j%M==0: #If reached coarse timepoint, then bring coarse path up to this point
                Sc+=sde(Sc,dtc,tc,dWc)#...Develop coarse path
                mc=min(mc,Sc) #Update minimum
                tc=tn #Coarse path now at j*T/Nsteps
                dtc=0
                dWc=0 #...Re-initialise coarse BI to 0

        Mf[num]=mf
        Mc[num]=mc
        Xf[num]=Sf
        Xc[num]=Sc
        num+=1 #One more simulation down

    return Xf,Mf,Xc,Mc

#######################################################################################################################
##Asset Plots
def diffusion_asset_plot(self,L=6,M=2):
    """
    Plots underlying asset price for diffusion with given sde function (EM) on a fine and coarse grid differing by factor
    of M.
    Modelling SDE:
    ~~ dS=mu(S,t)dt+sig(S,t)*dW~~

    Parameters:
        self(Option): option that function is called through
        L(int) : fine discretisation level 
        M(int) : coarseness factor s.t number of fine steps = M**L
    """
    T=self.T;X0=self.X0
    sde=self.sde

    Nsteps=M**L
    dt=T/Nsteps
    sqrt_dt=np.sqrt(dt)

    #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
    Xf=[X0];Xc=[X0]
    dWc=0
    for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
        t_=(j-1)*dt #Current time to simulate from in Ito calculus
        dWf=np.random.randn()*sqrt_dt
        dWc=dWc+dWf #Keep adding to coarse BI every loop until j is integer multiple of M
        Xf+=[Xf[-1] + sde(Xf[-1],dt,t_,dWf)]
        if j%M==0: #if j is integer multiple of M...
            Xc+=[Xc[-1] + sde(Xc[-1],M*dt,t_,dWc)] #...Develop coarse path
            dWc=0 #...Re-initialise coarse BI to 0

    ##Plot and label
    tf=np.linspace(0,T,Nsteps+1) #Fine time grid
    tc=np.linspace(0,T,M**(L-1)+1) #Coarse time grid
    plt.plot(tf,Xf,'k-',label='Fine')
    plt.plot(tc,Xc,'k--',label='Coarse')
    label=' '.join(str(type(self).__name__).split('_'))    
    plt.legend(framealpha=1,frameon=True)
    plt.title(label+f' Underlying Asset Price, $M={M}, L={L}$')
    plt.xlabel('$t$')
    plt.ylabel('Asset Price')

def JD_asset_plot(self,L=6,M=2):
    """
    Plots underlying asset price for general JD with given sde function (EM) on a fine and coarse grid differing by factor
    of M.

    Modelling SDE:
    ~~ S_=r*(Sn)*dt+sig(Sn,t)*dW+c(Sn,t)*[-lam*Jbar*dt] - term in square brackets to make process martingale
    ~~ Sn+1=S_+c(Sn,t)*dJ
    Idea:
    ___o_X_o__oX_o___X_o___X  | Coarse with jumps
    __Xo_X_oX_oX_oX__X_oX__X  | Fine with jumps
    -------------oX--X------  | Fine has to have fine timestep
    -------------o---X------  | Coarse can have longer increment here, but has to respect jumps

    Parameters:
        self(Option): option that function is called through
        L(int) : fine discretisation level 
        M(int) : coarseness factor s.t number of fine steps = M**L
    """

    X0=self.X0;lam=self.lam;T=self.T
    sde=self.sde
    Nsteps=M**L

    #Initialise asset price and time
    dWc=0;tau=0;tf=0;tc=0;dtc=0
    Xf=[X0];Xc=[X0];times_f=[0];times_c=[0]

    ##Algorithm Start
    tau+=self.jumptime(scale=1/lam)
    for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed

            tn=j*T/Nsteps #Fine timestepping right boundary

            while tau<tn: #If jump is before right boundary...
                dt=tau-tf #Adaptive step size is from last jump or left fine timestep
                dtc+=dt #Coarse timestep increments
                dWf=np.random.randn()*np.sqrt(dt) #Brownian motion for adaptive time step
                dWc+=dWf #Increment coarse BI
                dJ=np.exp(self.jumpsize())-1 #Generate jump

                #Develop fine path
                Xf+=[Xf[-1]+sde(Xf[-1],dt,tf,dWf,dJ)]

                #Develop coarse path
                Xc+=[Xc[-1]+sde(Xc[-1],dtc,tc,dWc,dJ)]

                dWc=0 #Reset coarse BI
                dtc=0 #Reset coarse timestep
                tf=tau #Both fine and coarse paths now at t_=latest jump time
                tc=tau
                times_f+=[tau]
                times_c+=[tau]
                tau+=self.jumptime(scale=1/lam) #Next jump time

            #Next jump time is after current right fine timestep
            dt=tn-tf #Adaptive time step is time from recent jump or left fine time up to right fine time
            dtc+=dt #Increment coarse timestep
            dWf=np.random.randn()*np.sqrt(dt) #Fine BI for adaptive timestep
            dWc+=dWf #Increment coarse BI
            Xf+=[Xf[-1]+sde(Xf[-1],dt,tf,dWf)] #Develope fine timestep
            tf=tn #Fine path now at j*T/Nsteps, set as left boundary
            times_f+=[tf]
            if j%M==0: #If reached coarse timepoint, then bring coarse path up to this point
                Xc+=[Xc[-1]+sde(Xc[-1],dtc,tc,dWc)]#...Develop coarse path
                tc=tn #Coarse path now at j*T/Nsteps
                times_c+=[tc]
                dtc=0
                dWc=0 #...Re-initialise coarse BI to 0

    ##Plot and label
    plt.plot(times_f,Xf,'k-',label='Fine')
    plt.plot(times_c,Xc,'k--',label='Coarse')
    plt.legend(framealpha=1, frameon=True)
    label=' '.join(str(type(self).__name__).split('_'))
    plt.title(label+f' Underlying Asset Price, $M={M}, L={L}$')
    plt.xlabel('$t$')
    plt.ylabel('Asset Price')
    
#######################################################################################################################
class Option:
    """
    Base class for all options.
    
    Attributes:
        alpha_0 (float) : weak order of convergence of option sde
        X0 (float) : Initial underlying asset price X(0) 
        r (float) : risk-free interest rate
        K (float) : Strike price (overridden and set to None for Lookback options)
        T (float) : Time to maturity for option
    Methods:
        __init__: Constructor
        payoff : payoff function for option type
        path : calculates path-wise quantities necessary to evaluate payoff
        sde : time-stepping function to develop underlying asset path
        looper : Interfaces with mlmc function to implement loop over Nl samples and generate payoff sums

    """
    def __init__(self,alpha_0=None,X0=100,K=100,T=1,r=0.05):
        """ 
        The Constructor for Option class. 
  
        Parameters: 
            alpha_0 (float) : weak order of convergence of option sde
            X0 (float) : Initial underlying asset price X(0) 
            r (float) : risk-free interest rate
            K (float) : Strike price (overridden and set to None for Lookback options)
            T (float) : Time to maturity for option 
        """
        self.alpha_0=alpha_0
        self.X0=X0
        self.r = r
        self.K = K
        self.T = T
        
    #Virtual functions to be overridden by specific sub-classes
    def payoff(self,N_loop,l,M): #Depends on option type
        """ 
        The payoff function for Option inheritor. Should call self.path function.
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
         """
        raise NotImplementedError("Option instance has no implemented payoff method")
        
    def anti_payoff(self,N_loop,l,M): #Depends on option type
        """ 
        The anti_payoff function for Option inheritor with antithetic estimators. Should call self.anti_path function.
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Pf,Pc (numpy.array) : payoff vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
         """
        raise NotImplementedError("Option instance has no implemented anti_payoff method")
        
    def path(self,N_loop,l,M): #Depends on underlying SDE
        """ 
        The path function for option inheritor, which calculates path-wise quantities. Should call self.sde function. 
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Pathwise_f,Pathwise_c (numpy.array) : pathwise quantity vectors for N_loop sample paths (Pc=Xc=X0 if l==0)
         """
        raise NotImplementedError("Option instance has no implemented path method")
        
    def anti_path(self,N_loop,l,M): #Depends on underlying SDE
        """ 
        The anti_path function for option inheritor, which calculates antithetic path-wise quantities. Should call self.sde function. 
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Pathwise_f,Pathwise_a,Pathwise_c (numpy.array) : pathwise quantity vectors for N_loop sample paths (Pa=Pf,Pc=Xc=X0 if l==0)
         """
        raise NotImplementedError("Option instance has no implemented anti_path method")
        
    def sde(self,X,dt,t_,dW,dJ=0): #Depends on underlying SDE
        """ 
        The sde time stepping function for option inheritor, which develops underlying asset path.
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
            X(np.array of floats): vector of asset prices at current time step for various sample paths
            dt(float) : size of time step
            t_(float) : current time
            dW(np.array of floats): same size as X, vector of Brownian increments
            dJ(np.array of floats): same size as X, vector of jumps s.t Xt=(1+dJ)*Xt-
        Returns:
            Xnew (numpy.array) : vector of asset prices at next time step
         """
        raise NotImplementedError("Option instance has no implemented sde method")
    
    def BS(self): #Depends on whether BS formula exists for option
        """ 
        The analytic Black-Scholes formula for the given Option instance, if it exists.
        No default. Should be implemented for any specific Option inheritors. 
  
        Parameters:
            self(Option): option that function is called through
        Returns:
            c (float) : analytic option price
         """
        raise NotImplementedError("Option instance has no implemented BS method")
    
    def asset_plot(self,L=6,M=2):
        """
        Plots underlying asset price for given sde function (EM) on a fine and coarse grid differing by factor of M.
        No default. Should be implemented for any specific Option inheritors. 

        Parameters:
            self(Option): option that function is called through
            L(int) : fine discretisation level 
            M(int) : coarseness factor s.t number of fine steps = M**L
        """
        raise NotImplementedError("Option instance has no implemented asset_plot method")
        
    #~~~Common functions to all sub-classes~~~#
    ##Interfaces with mlmc algorithm
    def looper(self,Nl,l,M,anti,Npl=10**4):
        """ 
        Interfaces with mlmc function to implement loop over Nl samples and generate payoff sums.
  
        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
            anti(bool) : whether to use antithetic estimator
            Npl(int) : size of sample path vector for each loop (i.e. number of samples per loop)
        Returns:
            suml (numpy.array) = [np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sum(Pf*Pc)]
            7d vector of various payoff sums and payoff-squared sums for Nl samples at level l/l-1
            Returns [sumPf,sumPf2,sumPf,sumPf2,0,0,0] is l=0.
         """
        num_rem=Nl #Initialise remaining samples for while loop
        suml=np.zeros(7)
        while (num_rem>0): #<---Parallelise this while loop
            N_loop=min(Npl,num_rem) #Break up Nl into manageable chunks of size Npl, until last iteration
            num_rem-=N_loop #On final iteration N_loop=num_rem, so num_rem will be=0 and loop terminates
            
            if anti==True: #Use antithetic estimators
                Pf,Pc=self.anti_payoff(N_loop,l,M)
            else:
                Pf,Pc=self.payoff(N_loop,l,M)
                
            sumPf=np.sum(Pf)
            sumPf2=np.sum(Pf**2)
            if l==0:
                suml+=np.array([sumPf,sumPf2,sumPf,sumPf2,0,0,0])
            else:
                dP_l=Pf-Pc #Payoff difference
                sumPc=np.sum(Pc)
                sumPc2=np.sum(Pc**2)
                sumPcPf=np.sum(Pc*Pf)
                suml+=np.array([np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sumPcPf])
                
        return suml 

    ##MLMC function
    def mlmc(self,eps,M=2,anti=False,N0=10**3, warm_start=True):
        """
        Runs MLMC method for given option (e.g. European) which returns an array of sums at each level.
        ________________
        Example usage:
        Euro=Euro_GBM()
        sums,N=Euro.mlmc(eps=0.1)
        ________________
        
        Parameters:
            self(Option) : Option instance (with SDE params and order of weak convergence of method alpha_0)
            eps(float) : desired accuracy
            M(int) = 2 : coarseness factor
            anti(bool) = False : whether to use antithetic estimator
            N0(int) = 10**3 : default number of samples to use when initialising new level
            warm_start(bool) = True: whether to save calculated alpha as alpha_0 for future function calls

        Returns: sums=[np.sum(dP_l),np.sum(dP_l**2),sumPf,sumPf2,sumPc,sumPc2,sum(Pf*Pc)],N
            sums(np.array) : sums of payoff diffs at each level and sum of payoffs at fine level, each column is a level
            N(np.array of ints) : final number of samples at each level
        """
        alpha_0=self.alpha_0
        alpha=alpha_0
        L=2

        V=np.zeros(L+1) #Initialise variance vector of each levels' variance
        N=np.zeros(L+1) #Initialise num. samples vector of each levels' num. samples
        dN=N0*np.ones(L+1) #Initialise additional samples for this iteration vector for each level
        sums=np.zeros((7,L+1)) #Initialise sums array, each column is a level
        sqrt_h=np.sqrt(M**(np.arange(0,L+1)))

        while (np.sum(dN)>0): #Loop until no additional samples asked for
            for l in range(L+1):
                num=dN[l]
                if num>0: #If asked for additional samples...
                    sums[:,l]+=self.looper(int(num),l,M,anti) #Call function which gives sums
            
            N+=dN #Increment samples taken counter for each level
            Yl=np.abs(sums[0,:])/N
            V=np.maximum((sums[1,:]/N)-(Yl)**2,0) #Calculate variance based on updated samples
            sqrt_V=np.sqrt(V)
            Nl_new=np.ceil((2*eps**-2)*np.sum(sqrt_V*sqrt_h)*(sqrt_V/sqrt_h)) #Estimate optimal number of samples at each level
            dN=np.maximum(0,Nl_new-N) #Number of additional samples

            if alpha_0==None: #Estimate order of weak convergence using LR
                #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^-l*alpha
                #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
                #This was actually incorrectly implemented by pefarrell until we corrected it!
                X=np.ones((L,2))
                X[:,0]=np.arange(1,L+1)
                a = np.linalg.lstsq(X,np.log(Yl[1:]),rcond=None)[0]
                alpha = max(0.5,-a[0]/np.log(M))
        
            if sum(dN > 0.01*N) == 0: #Almost converged
                if max(Yl[-2]/(M**alpha),Yl[-1])>(M**alpha-1)*eps*np.sqrt(0.5):
                    L+=1
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=np.concatenate((V,np.zeros(1)), axis=0)
                    N=np.concatenate((N,N0*np.zeros(1)),axis=0)
                    dN=np.concatenate((dN,N0*np.ones(1)),axis=0)
                    sqrt_h=np.concatenate((sqrt_h,[np.sqrt(M**L)]),axis=0)
                    sums=np.concatenate((sums,np.zeros((7,1))),axis=1)
                    
        print(f'Estimated alpha = {alpha}')
        if warm_start:
            self.alpha_0=alpha #update with estimate of option alpha
            print(f'    Saved estimated alpha_0 = {alpha}')
        return sums,N

class JumpDiffusion_Option(Option):
    """
    Base class for general Jump Diffusion Options. Inherits from Option class. Lacks implemented path, payoff methods.
    ___________
    S_=r*(Sn)*dt+sig(Sn,t)*dW+c(Sn,t)*[-lam*Jbar*dt] - term in square brackets to make process martingale
    Sn+1=S_+c(Sn,t)*dJ
    
    S_t+h=Y*S_t => dS=S_t+h-S_t=(Y-1)*S_t
    Q=ln(Y)~N(a,b) for example
    dJ=exp(Q)-1 #Size of jump
    Jbar=E[exp(Q)-1] #Expected jump size
    __________
    Attributes:
        lam(float) : expected number of jumps per unit time
        jumpsize(func<>) : rng s.t dJ=exp(jumpsize())-1
        jumptime(func<(float)>) : rng s.t t_jump_n+1-t_jump_n ~ jumptime(lam)
        J_bar(float) : E[exp(jumpsize)-1] Expected value of dJ
        sig(func<(float),(float)>) : volatility of asset as function of asset price x, time t; scales dW
        c(func<(float),(float)>) : function of asset price x, time t; scales dJ
        __Inherited__
    """
    asset_plot = JD_asset_plot
    def __init__(self,payoff,path,lam=1,jumpsize=np.random.standard_normal,jumptime=np.random.exponential,
                 J_bar=None,sig=lambda x,t:0.2*x,c=lambda x,t: x,**kwargs):
        """
        Constructor for JumpDifusion_Option class. Passes **kwargs to inherited Option constructor.

        Parameters:
                payoff(func) : desired payoff function
                path(func) : desired jump diffusion path function
                lam(float) : expected number of jumps per unit time
                jumpsize(func<>) : rng s.t dJ=exp(jumpsize())-1
                jumptime(func<(float)>) : rng s.t t_jump_n+1-t_jump_n ~ jumptime(lam)
                J_bar(float) : E[exp(jumpsize)-1] Expected value of dJ
                sig(func<(float),(float)>) : volatility of asset as function of asset price x, time t; scales dW
                c(func<(float),(float)>) : function of asset price x, time t; scales dJ
                __**kwargs__
                    alpha_0 (float) : weak order of convergence of option sde
                    X0 (float) : Initial underlying asset price X(0) 
                    r (float) : risk-free interest rate
                    K (float) : Strike price (overridden and set to None for Lookback options)
                    T (float) : Time to maturity for option
        """
        super().__init__(**kwargs)
        self.path=WeakMethod(path,self)
        self.payoff=WeakMethod(payoff,self)
        self.lam=lam
        if J_bar==None:
            if jumpsize!=np.random.standard_normal:
                raise ValueError("If specifying random distribution for Q, specify Jbar=E[exp(Q)-1].")
            else:
                self.J_bar=np.exp(0.5)-1
        else:
            self.J_bar=J_bar
        self.jumpsize=jumpsize
        self.jumptime=jumptime
        self.sig=sig
        self.c=c
        
    def sde(self,X,dt,t_,dW,dJ=0):
        """ 
        The Euler-Maruyama time stepping function for MJD, which develops underlying asset path.
        S_=r*(Sn)*dt+sig(Sn,t)*dW+c(Sn,t)*[-lam*Jbar*dt] - term in square brackets to make process martingale
        Sn+1=S_+c(Sn,t)*dJ
  
        Parameters:
            self(Option): option that function is called through
            X(np.array of floats): vector of asset prices at current time step for various sample paths
            dt(float) : size of time step
            t_(float) : current time
            dW(np.array of floats): same size as X, vector of Brownian increments
            dJ(np.array of floats): same size as X, vector of jump sizes - default 0
        Returns:
            Xnew (numpy.array) : vector of asset prices at next time step
         """
        dx_=self.r*X*dt + self.sig(X,t_)*dW + self.c(X,t_)*(-self.lam*self.J_bar*dt)
        return dx_+self.c(X+dx_,t_+dt)*dJ

class Diffusion_Option(Option):
    """
    Base class for general Diffusion Options. Inherits from Option class. Lacks implemented path, payoff methods.
    ___________
    dS=(r*S+drift(S,t))dt+sig(S,t)*dW
    __________
    Attributes:
        drift(func<(float),(float)>) : drift term, function of asset price x, time t; scales dt
        sig(func<(float),(float)>) : volatility of asset as function of asset price x, time t; scales dW
        __Inherited__
    """
    asset_plot = diffusion_asset_plot
    def __init__(self,path,payoff,drift=lambda x,t: 0,sig=lambda x,t:0.2*x,**kwargs):
        """
        Constructor for Diffusion_Option class. Passes **kwargs to inherited Option constructor.
        
        Parameters:
            path(func) : desired jump diffusion path function
            payoff(func) : desired payoff function
            drift(func<(float),(float)>) : drift term, function of asset price x, time t; scales dt
            sig(func<(float),(float)>) : volatility of asset as function of asset price x, time t; scales dW
            __**kwargs__
                alpha_0 (float) : weak order of convergence of option sde
                X0 (float) : Initial underlying asset price X(0) 
                r (float) : risk-free interest rate
                K (float) : Strike price (overridden and set to None for Lookback options)
                T (float) : Time to maturity for option
        """
        super().__init__(**kwargs)
        self.path=WeakMethod(path,self)
        self.payoff=WeakMethod(payoff,self)
        self.sig = sig
        self.mu = lambda x,t: self.r*x+drift(x,t) # mu(S,t)=(r*S+drift(S,t))

    def sde(self,X,dt,t_,dW):
        """ 
        The Euler-Maruyama time stepping function for general Diffusion, which develops underlying asset path.
        dS = mu(S,t)dt+sig(S,t)*dW
  
        Parameters:
            self(Option): option that function is called through
            X(np.array of floats): vector of asset prices at current time step for various sample paths
            dt(float) : size of time step
            t_(float) : current time
            dW(np.array of floats): same size as X, vector of Brownian increments
        Returns:
            Xnew (numpy.array) : vector of asset prices at next time step
         """
        return self.mu(X,t_)*dt+self.sig(X,t_)*dW
    
    def asset_plot(self,L=6,M=2):
        """
        Plots underlying asset price for general Diffusion with given sde function (EM) on a fine and coarse grid differing by factor of M.
        Modelling SDE:
        ~~ dS=mu(S,t)dt+sig(S,t)*dW~~
        
        Parameters:
            self(Option): option that function is called through
            L(int) : fine discretisation level 
            M(int) : coarseness factor s.t number of fine steps = M**L
        """
        T=self.T;X0=self.X0
        sde=self.sde

        Nsteps=M**l
        dt=T/Nsteps
        sqrt_dt=np.sqrt(dt)

        #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
        Xf=[X0];Xc=[X0]
        dWc=0
        for j in range(1,Nsteps+1): #Note that if Nsteps=1 (l=0), j=1 and so coarse path not developed
            t_=(j-1)*dt #Current time to simulate from in Ito calculus
            dWf=np.random.randn()*sqrt_dt
            dWc=dWc+dWf #Keep adding to coarse BI every loop until j is integer multiple of M
            Xf+=[Xf[-1] + sde(Xf[-1],dt,t_,dWf)]
            if j%M==0: #if j is integer multiple of M...
                Xc+=[Xc[-1] + sde(Xc[-1],M*dt,t_,dWc)] #...Develop coarse path
                dWc=0 #...Re-initialise coarse BI to 0
                
        ##Plot and label
        tf=np.arange(0,T+dt,dt) #Fine time grid
        tc=np.arange(0,T+M*dt,M*dt)  #Coarse time grid
        plt.plot(tf,Xf,label='Fine')
        plt.plot(tc,Xc,label='Coarse')
        plt.title(f'Diffusion Model Underlying Asset Price, $M={M}, L={L}$')
        plt.xlabel('$T$')
        plt.ylabel('Asset Price')

class MyOption(Option):
    """
    Base class for general options. Inherits from Option class. Can set sde, path, payoff via constructor.
    
    Attributes:
        sde(func) : SDE goevrning underlying asset evolution
        path(func) : path-wise calculation of relevant quantities for payoff
        payoff(func) : relevant payoff function
        ~~ + extra attributes necessary for path/sde/payoff methods ~~
        __Inherited__

    """
    def __init__(self,sde,path,payoff,**kwargs):
        """
        Constructor for MyOption class. Can set sde, path, payoff as args.

        Parameters:
            sde(func) : SDE goevrning underlying asset evolution
            path(func) : path-wise calculation of relevant quantities for payoff
            payoff(func) : relevant payoff function
            __**kwargs__
                alpha_0 (float) : weak order of convergence of option sde
                X0 (float) : Initial underlying asset price X(0) 
                r (float) : risk-free interest rate
                K (float) : Strike price (overridden and set to None for Lookback options)
                T (float) : Time to maturity for option
                ~~ + extra kwargs necessary for path/sde/payoff methods ~~
        """
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.sde=WeakMethod(sde,self)
        self.path=WeakMethod(path,self)
        self.payoff=WeakMethod(payoff,self)
        
#######################################################################################################################
#Specific option implementations
class Merton_Option(Option):
    """
    Base class for Merton Jump Diffusion Options. Inherits from Option class. Lacks implemented path method.
    ___________
    S_=r*(Sn)*dt+sig(Sn,t)*dW+c(Sn,t)*[-lam*Jbar*dt] - term in square brackets to make process martingale
    Sn+1=S_+c(Sn,t)*dJ
    
    S_t+h=Y*S_t => dS=S_t+h-S_t=(Y-1)*S_t
    Q=ln(Y)~N(a,b)
    dJ=exp(Q)-1 #Size of jump
    Jbar=E[exp(Q)-1] #Expected jump size
    __________
    Attributes:
        lam(float) : expected number of jumps per unit time
        jumpsize(func<>) : N(jumpmean,jumpstd**2) s.t dJ=exp(jumpsize)-1
        jumptime(func<>) : Exponential process s.t jumptime ~ Exp(lam)
        jumpmean(float) : mean of lognormal jumpsize s.t. log(dJ)~N(jumpmean,jumpstd**2)
        jumpstd(float) : std of lognormal jumpsize s.t. log(dJ)~N(jumpmean,jumpstd**2)
        J_bar(float) : E[exp(jumpsize)-1] Expected value of dJ
        sig(float) : volatility of asset as function; scales X*dW
        __Inherited__
    """ 
    
    asset_plot = JD_asset_plot
    
    def __init__(self,lam=1,jumpmean=0.1,jumpstd=0.2,sig=0.2,**kwargs):
        """
        Constructor for Merton_Option class. Passes **kwargs to inherited Option constructor.

        Parameters:
                lam(float) : expected number of jumps per unit time
                jumpmean(float) : mean of lognormal jumpsize s.t. log(dJ)~N(jumpmean,jumpstd)
                jumpstd(float) : std of lognormal jumpsize s.t. log(dJ)~N(jumpmean,jumpstd)
                sig(float) : constant volatility of asset ; scales dW
                __**kwargs__
                    alpha_0 (float) : weak order of convergence of option sde
                    X0 (float) : Initial underlying asset price X(0) 
                    r (float) : risk-free interest rate
                    K (float) : Strike price (overridden and set to None for Lookback options)
                    T (float) : Time to maturity for option
        """
        super().__init__(**kwargs)
        self.lam=lam
        self.J_bar = np.exp(jumpmean+0.5*jumpstd**2)-1
        self.jumpstd = jumpstd
        self.jumpmean = jumpmean
        self.jumptime = np.random.exponential
        self.jumpsize = lambda : jumpmean+jumpstd*np.random.standard_normal()
        self.sig=sig
        
    def sde(self,X,dt,t_,dW,dJ=0):
        """ 
        The Euler-Maruyama time stepping function for MJD, which develops underlying asset path.
        S_=r*(Sn)*dt+sig*Sn*dW+Sn*[-lam*Jbar*dt] - term in square brackets to make process martingale
        Sn+1=S_+Sn*dJ
  
        Parameters:
            self(Option): option that function is called through
            X(np.array of floats): vector of asset prices at current time step for various sample paths
            dt(float) : size of time step
            t_(float) : current time
            dW(np.array of floats): same size as X, vector of Brownian increments
            dJ(np.array of floats): same size as X, vector of jump sizes - default 0
        Returns:
            Xnew (numpy.array) : vector of asset prices at next time step
         """
        dx_=(self.r-self.lam*self.J_bar)*X*dt + self.sig*X*dW 
        return dx_+(dx_+X)*dJ
    
class GBM_Option(Option):
    """
    Base class for Geometric Brownian Motion Options. Inherits from Option class. Lacks implemented path method.
    __________________
    dS=mu*S*dt+sig*S*dW, mu = drift+r
    __________________
    
    Attributes:
        sig(float) : constant volatility of underlying asset
        drift(float) : drift term (default 0)
        __Inherited__
    """
    asset_plot=diffusion_asset_plot
    def __init__(self,drift=0,sig=0.2,**kwargs):
        """ 
        The Constructor for GBM_Option class. Passes *kwargs to Option Constructor.
  
        Parameters:
            sig(float) : constant volatility of underlying asset
            drift(float) : drift term (default 0)
            __**kwargs__
                alpha_0 (float) : weak order of convergence of option sde
                X0 (float) : Initial underlying asset price X(0) 
                r (float) : risk-free interest rate
                K (float) : Strike price (overridden and set to None for Lookback options)
                T (float) : Time to maturity for option 
        """
        super().__init__(**kwargs)
        self.sig = sig
        r=self.r
        self.mu = (r+drift)

    def sde(self,X,dt,t_,dW):
        """ 
        The Euler-Maruyama time stepping function for GBM, which develops underlying asset path.
        dS=r*S*dt+sig*S*dW
  
        Parameters:
            self(Option): option that function is called through
            X(np.array of floats): vector of asset prices at current time step for various sample paths
            dt(float) : size of time step
            t_(float) : current time
            dW(np.array of floats): same size as X, vector of Brownian increments
        Returns:
            Xnew (numpy.array) : vector of asset prices at next time step
         """
        return self.mu*X*dt+self.sig*X*dW

#######################################################################################################################
##GBM Implementations
class Euro_GBM(GBM_Option):
    """
    Class for Geometric Brownian Motion European Call Options. Inherits from GBM_Option class.
    __________________
    payoff=max(S(T)-K,0)
    __________________
    
    Attributes:
        __Inherited__
    """
    
    payoff = EA_payoff #Set payoff method to European call option payoff
    anti_payoff = anti_EA_payoff #Set anti_payoff method to European call option payoff with antithetic estimator
    path = diffusion_path #Set path to diffusion path which calculates final asset prices
    anti_path = diffusion_anti_path #Set anti_path to diffusion path to calculates final asset prices with antithetics
    
    def BS(self):
        """
        Black scholes formula for European Call with GBM.
        
        Returns:
            c(float) : Black-Scholes option price for given European option instance
        """
        D1 =(np.log(self.X0/self.K)+(self.r+0.5*self.sig**2)*self.T)/(self.sig*np.sqrt(self.T))
        D2 = D1 - self.sig*np.sqrt(self.T)
        c=self.X0*norm.cdf(D1)-self.K*np.exp(-self.r*self.T)*norm.cdf(D2)
        return c
    
class Asian_GBM(GBM_Option):
    """
    Class for Geometric Brownian Motion Asian Call Options. Inherits from GBM_Option class.
    __________________
    payoff=max(avg(S)-K,0)
    __________________
    
    Attributes:
        __Inherited__
    """
    payoff = EA_payoff #Set payoff method to European call option payoff
    anti_payoff = anti_EA_payoff #Set anti_payoff method to European call option payoff with antithetic estimator
    path = diffusion_path_avg #Set as diffusion path which calculates final asset prices avgs
    anti_path = diffusion_anti_path_avg #Set as diffusion path to calculate final asset prices avgs with antithetics
    
    
    def BS(self):
        """
        Black scholes formula for Asian Call with GBM.
        Formula from: 
        Turnbull and Wakeman (1991), “A Quick Algorithm for Pricing European Average Option,” Journal of 
        Financial and Quantitative Analysis 26, pp. 377–389.
        See also: http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch10%20Asian%20Options.pdf

        Returns:
            c(float) : Black-Scholes option price for given Asian option instance
        """
        X0=self.X0;r=self.r;T=self.T;sig=self.sig;K=self.K
        M1 = X0*(np.exp(r*T)-1)/(r*T) #=E[average]
        M2 = (2*X0**2)*(np.exp((2*r+sig**2)*T)/((r+sig**2)*(2*r+sig**2)*T**2)+
        (1/(2*r+sig**2)-np.exp(r*T)/(r+sig**2))/(r*T**2))
        sig_A=np.sqrt(np.log(M2/(M1**2)))
        d1=(np.log(M1/K)+0.5*sig_A**2)/sig_A
        d2=d1-sig_A
        return np.exp(-r*T)*(M1*norm.cdf(d1)-K*norm.cdf(d2))
        
class Lookback_GBM(GBM_Option):
    """
    Class for Geometric Brownian Motion Lookback Call Options. Inherits from GBM_Option class.
    __________________
    payoff=max(S(T)-min(S),0)
    __________________
    
    Attributes:
        beta(float) = 0.5826 : Special factor for offset correction
        __Inherited__
    """
    payoff=Lookback_payoff #Set payoff method to Lookback call payoff
    beta=0.5826 #Special factor for offset correction
    path = diffusion_path_min #Set path method to path whih returns Xf,Mf,Xc,Mc
    
    def BS(self):
        """
        Black scholes formula for Lookback Call with GBM.
        
        Returns:
            c(float) : Black-Scholes option price for given Lookback option instance
        """
        D1 =(self.r+0.5*self.sig**2)*self.T/(self.sig*np.sqrt(self.T))
        D2 = D1 - self.sig*np.sqrt(self.T)
        k   = 0.5*self.sig**2/self.r;
        return self.X0*(norm.cdf(D1) - norm.cdf(-D1)*k - np.exp(-self.r*self.T)*(norm.cdf(D2) - norm.cdf(D2)*k) )
        
class Digital_GBM(GBM_Option):
    """
    Class for Geometric Brownian Motion Digital Call Options. Inherits from GBM_Option class.
    __________________
    payoff=K*H(S(T)-K)
    H is Heaviside step function
    __________________
    
    Attributes:
        __Inherited__
    """
    
    payoff = Digital_payoff #Set payoff method to Digital Call option payoff
    path = diffusion_path
    
    def BS(self):
        """
        Black scholes formula for Digital Call with GBM.
        
        Returns:
            c(float) : Black-Scholes option price for given Digital option instance
        """
        D2 = (np.log(self.X0/self.K)+(self.r+0.5*self.sig**2)*self.T)/(self.sig*np.sqrt(self.T)) - self.sig*np.sqrt(self.T)
        return self.K*np.exp(-self.r*self.T)*norm.cdf(D2)

class Amer_GBM(GBM_Option):
    """
    Class for Geometric Brownian Motion American Put Options. Inherits from GBM_Option class.
    __________________
    payoff = Amer_payoff
    __________________
    
    Attributes:
        exerciseBoundary(func<Option,float>) : exercise boundary function for option
        __Inherited__
    """
    payoff=Amer_payoff
    def __init__(self,exerciseBoundary,**kwargs):
        super().__init__(**kwargs)
        self.exerciseBoundary = exerciseBoundary

    def exerciseBoundary(self,t):
        """ 
        The exercise boundary function for parent American Option.

        Parameters:
            self(Option): option that function is called through
            t(float) : time at which exercise boundary function evaluated
        Returns:
            b (float) : value of exercise boundary at time t
        """
        raise NotImplementedError("American Option instance has no implemented exerciseBoundary method") 
    
    def path(self,N_loop,l,M):
        """ 
        The path function for Euler-Maruyama GBM, which calculates path-wise values for American option pricing.

        Parameters:
            self(Option): option that function is called through
            N_loop(int): total number of sample paths to evaluate payoff on
            l(int) : discretisation level
            M(int) : coarseness factor, number of fine steps = M**l
        Returns:
            Xf,Xc (numpy.array) : N-by-3 arrays s.t X[:,0]=final asset price, X[:,1] = final crossing probability, 
                                  X[:,2] = expected payoff if barrier hit.
        """
        if M!=2:
            raise ValueError("M must be equal to 2 for American Option pricer. Please use M=2.")
        T=self.T;X0=self.X0;sig=self.sig;K=self.K;r=self.r
        sde=self.sde;exerciseBoundary=self.exerciseBoundary
        Nsteps=M**l
        dt=T/Nsteps
        sqrt_dt=np.sqrt(dt)

        #Initialise fine, coarse asset prices; coarse Brownian increment (BI)
        dWc=np.zeros(N_loop)
        Xf = np.zeros((N_loop, 3))

        # path variables X_t 
        Xf[:,0] = X0* np.ones(N_loop)

        # Probability of not having crossed yet
        Xf[:,1] = np.ones(N_loop)

        # Payoff part 1 --> barrier hit during some interval
        Xf[:,2] = np.zeros(N_loop)
        Xc = Xf.copy()

        for j in range(1,Nsteps+1):
            t_ = (j-1)*dt
            dWf = sqrt_dt*np.random.randn(N_loop)
            dWc += dWf

            Xleft = Xf[:,0]

            # advance path variable
            Xright = Xleft + sde(Xleft,dt,t_,dWf)

            # evaluate barrier on left end point
            leftBarrier = exerciseBoundary(self,t_)
            tfMid = t_ + dt/2

            # evaluate barrier on right end point 
            rightBarrier = exerciseBoundary(self,j*dt)

            # evaluate barrier at mid point
            midBarrier = exerciseBoundary(self,tfMid)

            Prob1 = np.exp(-2 * np.maximum(Xleft - leftBarrier, 0) * np.maximum(Xright - rightBarrier,0) / ((sig**2)*dt*Xleft**2))

            # update payoff        
            Xf[:,2] += Xf[:,1]*Prob1 *np.maximum(K-midBarrier,0)*np.exp(-r*(tfMid))
            # update crossing probability
            Xf[:,1] *= (1.0-Prob1)
            # update path value
            Xf[:,0] = Xright

            if j%M==M/2:
                #If at midpoint between coarse times, store dWc for calculation of Xmid
                halfdWc=dWc

            if j%M==0:
                t_=(j-M)*dt
                Xleft = Xc[:,0]

                # advance path variable
                Xright = Xleft + sde(Xleft,M*dt,t_,dWc)
                Xmid = Xleft + 0.5*(Xright-Xleft) + sig*Xleft*(halfdWc - 0.5*dWc)

                # evaluate barrier at left end point
                leftBarrier = exerciseBoundary(self,t_)

                tcMid = t_ + M*dt/2

                # evaluate barrier at right end point
                rightBarrier = exerciseBoundary(self,j*dt)
                
                # evaluate barrier at mid point
                midBarrier2 = exerciseBoundary(self,tcMid)

                # prob of hitting the barrier
                midBarrier1 = 0.5*(rightBarrier+leftBarrier)
                Prob11 = np.exp(-2*np.maximum(Xleft-leftBarrier,0) * np.maximum(Xmid-midBarrier1,0) /(dt*(sig**2)*(Xleft**2)))
                Prob12 = np.exp(-2*np.maximum(Xmid-midBarrier1,0) * np.maximum(Xright-rightBarrier,0)/(dt*(sig**2)*(Xleft**2)))

                Prob1 = (1.0-(1.0-Prob11)*(1.0-Prob12))
                # update payoff
                Xc[:,2] += Xc[:,1]*Prob1*np.maximum(K-midBarrier2,0)*np.exp(-r*tcMid)
                # update crossing probability
                Xc[:,1] *= (1-Prob1)
                # update path variable
                Xc[:,0] = Xright
                
                #Reset coarse Brownian increments
                dWc=np.zeros(N_loop) #...Re-initialise coarse BI to 0


        return Xf,Xc
    
    def asset_plot(self,L=6,M=2):
        """
        Plots underlying asset price for GBM with given sde function (EM) on a fine and coarse grid differing by factor
        of M, and exercise boundary for parent American Option.
        Modelling SDE:
        ~~ dS=(r+drift)*S*dt+sig*S*dW~~

        Parameters:
            self(Option): option that function is called through
            L(int) : fine discretisation level 
            M(int) : coarseness factor s.t number of fine steps = M**L
        """
        super().asset_plot(L,M)
        times=np.linspace(0,self.T)
        plt.plot(times, self.exerciseBoundary(self,times),'k:',label='Ex. boundary')
        plt.legend(framealpha=1,frameon=True)
        
#######################################################################################################################
##Merton Model Implementations
class Euro_Merton(Merton_Option):
    """
    Class for European Call Merton Jump Diffusion Options. Inherits from Merton_Option class.
    ___________
    payoff=max(S(T)-K,0)
    __________
    Attributes:
        __Inherited__
    """
    payoff = EA_payoff #Set payoff method to European Call option payoff
    path = JD_path
    
    def BS(self,n=100):
        """
        Approximate BS formula for European Call with MJD, truncate sum to n terms:
        
        Parameters:
            n(int)=100 : number of terms to truncate infinite sum to
        Returns:
            c(float) : BS price for Euro Call under MJD
        """
        sig_n=np.sqrt(self.sig**2+(self.jumpstd**2)*np.arange(0,n)/self.T)
        r_n=self.r-self.lam*self.J_bar+np.arange(0,n)*np.log((1+self.J_bar)/self.T)
        D1 =(np.log(self.X0/self.K)+(r_n+0.5*sig_n**2)*self.T)/(sig_n*np.sqrt(self.T))
        D2 = D1 - sig_n*np.sqrt(self.T)
        bs=self.X0*norm.cdf(D1)-self.K*np.exp(-r_n*self.T)*norm.cdf(D2)
        discount=np.exp(-self.lam*(1+self.J_bar)*self.T)
        c = discount*np.sum(((self.lam*(1+self.J_bar)*self.T)**np.arange(0,n)/factorial(np.arange(0,n)))*bs)
        return c

class Asian_Merton(Merton_Option):
    """
    Class for Asian Call Merton Jump Diffusion Options. Inherits from Merton_Option class.
    ___________
    payoff=max(avg(S)-K,0)
    __________
    Attributes:
        __Inherited__
    """
    
    payoff = EA_payoff #Set payoff method to Euro/Asian Call Option Payoff
    path = JD_path_avg

class Lookback_Merton(Merton_Option):
    """
    Class for Lookback Call Merton Jump Diffusion Options. Inherits from Merton_Option class.
    ___________
    payoff=max(S(T)-min(S),0)
    __________
    Attributes:
        beta(float)=0.5826 : Special offset for calculating min
        __Inherited__
    """
    
    payoff = Lookback_payoff #Set payoff method to Lookback Call Option Payoff
    beta = 0.5826 #Special offset for calculating min
    path = JD_path_min
    
class Digital_Merton(Merton_Option):
    """
    Class for Digital Call Merton Jump Diffusion Options. Inherits from Merton_Option class.
    ___________
    payoff=max(avg(S)-K,0)
    __________
    Attributes:
        __Inherited__
    """
    payoff = Digital_payoff
    path = JD_path

#######################################################################################################################
##Two plotting functions to plot Giles-style plots
def Giles_plot(option,eps,markers,label,fig,M=2,N0=10**3,anti=False,Lmax=8,Nsamples=10**6):
    """
    Plots variance/mean and cost/number of levels plots a la Giles 2008.
    
    Example Usage:
    Euro=Euro_GBM()
    variance_plot(Euro,0.005,label='European Call GBM ')
    
    Parameters:
        option(Option) : Option instance to call mlmc through
        eps(list-like) : desired accuracy of MLMC
        label(str) : plot title
        fig : figure to plot onto
        M(int) = 2 : coarseness factor
        N0(int) = 10**3 : min samples per level
        anti(bool) = False : whether to use antithetic estimator
        Lmax(int) = 8 : max level to estimate variance/mean of P_l-P_l-1
        Nsamples(int) = 10**6 : number of samples to use to estimate variance/mean
    """
    #Set plotting params
    markersize=3*(fig.get_size_inches()[0])/4
    if len(eps)!=len(markers):
        raise ValueError("Length of markers argument must be same as length of epsilon argument.")
    axis_list=fig.axes
    if len(axis_list)!=4:
        print('Expected 4 subplots in fig, attempting to proceed but may fail.')
    
    #Initialise complexity lists
    cost_mlmc=[]
    cost_mc=[]
    cost_anti=[]
    
    #Do the calculations and simulations for num levels and complexity plot
    for i in range(len(eps)):
        e=eps[i]
        sums,N=option.mlmc(e,M,warm_start=False,N0=N0)
        L=len(N)-1
        means_p=np.abs(sums[2,:]/N)
        V_p=(sums[3,:]/N)-means_p**2
        
        #Note that cost is defined as calls to rng (i.e. number of fine steps) not number of sde evaluations
        if hasattr(option,'lam'): #If jump diffusion option, have to add extra cost due to jumps
            cost_mlmc+=[(np.sum(N)*option.lam
                             +np.sum(N*(M**np.arange(0,L+1))))*e**2]
        else:
            cost_mlmc+=[np.sum(N*(M**np.arange(0,L+1)))*e**2]

        cost_mc+=[2*V_p[-1]*(M**L)]
        axis_list[2].semilogy(range(L+1),N,'k-',marker=markers[i],label=f'{e}',markersize=markersize,
                       markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
        
        if anti==True:
            sums,N=option.mlmc(e,M,anti=anti,warm_start=False,N0=N0)
            L=len(N)-1
            
            #Note that cost is defined as calls to rng (i.e. number of fine steps) not number of sde evaluations
            if hasattr(option,'lam'): #If jump diffusion option, have to add extra cost due to jumps
                cost_anti+=[(np.sum(N)*option.lam
                             +np.sum(N*(M**np.arange(0,L+1))))*e**2]
            else:
                cost_anti+=[np.sum(N*(M**np.arange(0,L+1)))*e**2]
                
            axis_list[2].semilogy(range(L+1),N,'k--',marker=markers[i],markersize=markersize,
                       markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
    
    #Variance and mean samples
    sums=np.zeros((4,Lmax+1))
    for l in range(Lmax+1):
        sums[:,l] = option.looper(Nsamples,l,M,anti=False)[:4]
    
    means_p=np.abs(sums[2,:]/Nsamples)
    V_p=(sums[3,:]/Nsamples)-means_p**2 
    means_dp=np.abs(sums[0,:]/Nsamples)
    V_dp=(sums[1,:]/Nsamples)-means_dp**2  
    
    #Plot variances
    axis_list[0].plot(range(Lmax+1),np.log(V_p)/np.log(M),'k:',label='$P_{l}$',
                      marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
    axis_list[0].plot(range(1,Lmax+1),np.log(V_dp[1:])/np.log(M),'k-',label='$P_{l}-P_{l-1}$',
                      marker=(8,2,0), markersize=markersize, markerfacecolor="None", markeredgecolor='k',
                      markeredgewidth=1)
    #Plot means
    axis_list[1].plot(range(Lmax+1),np.log(means_p)/np.log(M),'k:',label='$P_{l}$',
                      marker=(8,2,0), markersize=markersize, markerfacecolor="None",markeredgecolor='k',
                      markeredgewidth=1)
    axis_list[1].plot(range(1,Lmax+1),np.log(means_dp[1:])/np.log(M),'k-',label='$P_{l}-P_{l-1}$',
                      marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1)
                      
    #Plot antithetic means and variance if necessary
    if anti==True:
        for l in range(Lmax+1):
            sums[:,l] = option.looper(Nsamples,l,M,anti=True)[:4]

        means_p=np.abs(sums[2,:]/Nsamples)
        V_p=(sums[3,:]/Nsamples)-means_p**2 
        means_dp=np.abs(sums[0,:]/Nsamples)
        V_dp=(sums[1,:]/Nsamples)-means_dp**2

        #Plot antithetic variances
        axis_list[0].plot(range(1,Lmax+1),np.log(V_dp[1:])/np.log(M),'k--',label='Anti $P_{l}-P_{l-1}$',
                          marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k',
                          markeredgewidth=1)
        #Plot antithetic means
        axis_list[1].plot(range(1,Lmax+1),np.log(means_dp[1:])/np.log(M),'k--',label='Anti $P_{l}-P_{l-1}$',
                          marker=(8,2,0),markersize=markersize,markerfacecolor="None",markeredgecolor='k',
                          markeredgewidth=1)

    #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
    #Will use antithetic means/variances if anti==True
    X=np.ones((Lmax,2))
    X[:,0]=np.arange(1,Lmax+1)
    a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
    alpha = -a[0]/np.log(M)
    b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
    beta = -b[0]/np.log(M) 
    
    #Label variance plot
    axis_list[0].set_xlabel('$l$')
    axis_list[0].set_ylabel(f'log$_{M}$(var)')
    axis_list[0].legend(framealpha=1, frameon=True)
    axis_list[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #Add estimated beta
    if anti==True:
        s='$\\beta_{anti}$ = %s' % round(beta,2)
    else:
        s='$\\beta$ = {}'.format(round(beta,2))
    t = axis_list[0].annotate(s, (Lmax/2, np.log(V_dp[1])/np.log(M)),fontsize=markersize,
            size=2*markersize, bbox=dict(ec='None',facecolor='None',lw=2))
    
    #Label means plot
    axis_list[1].set_xlabel('$l$')
    axis_list[1].set_ylabel(f'log$_{M}$(mean)')
    axis_list[1].legend(framealpha=1, frameon=True)
    axis_list[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #Add estimated alpha
    if anti==True:
        s='$\\alpha_{anti}$ = %s' % round(alpha,2)
    else:
        s='$\\alpha$ = {}'.format(round(alpha,2))
    t = axis_list[1].annotate(s, (Lmax/2, np.log(means_dp[1])/np.log(M)), fontsize=markersize,
            size=2*markersize, bbox=dict(ec='None',facecolor='None',lw=2))
    
    #Label number of levels plot
    axis_list[2].set_xlabel('$l$')
    axis_list[2].set_ylabel('$N_l$')
    xa=axis_list[2].xaxis
    xa.set_major_locator(ticker.MaxNLocator(integer=True))
    (lines,labels)=axis_list[2].get_legend_handles_labels()
    ncol=1
    if anti==True:
        #Add to indicate antithetic
        ncol=2
        labels+=['Std. MLMC','Anti MLMC']
        lines += [plt.Line2D([], [], linestyle='-',color='k',label='Std. MLMC'),
                    plt.Line2D([], [], linestyle='--',color='k', label='Anti MLMC')]
        for i in range(len(eps)-2):
            lines.append(plt.Line2D([],[], alpha=0))
            labels.append('')

    leg = Legend(axis_list[2], lines, labels, ncol=ncol, title='Accuracy $\epsilon$',
                 frameon=True, framealpha=1)
    leg._legend_box.align = "left"
    axis_list[2].add_artist(leg)
        
    
    #Label and plot complexity plot
    axis_list[3].loglog(eps,cost_mc,'k:',marker=(8,2,0),markersize=markersize,
                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MC')
    axis_list[3].loglog(eps,cost_mlmc,'k-',marker=(8,2,0),markersize=markersize,
                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Std. MLMC')
    if anti==True:
        #Plot antithetic complexity
        axis_list[3].loglog(eps,cost_anti,'k--',marker=(8,2,0),markersize=markersize,
                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Anti MLMC')
    axis_list[3].set_xlabel('$\epsilon$')
    axis_list[3].set_ylabel('$\epsilon^{2}$cost')
    axis_list[3].legend(frameon=True,framealpha=1)
    
    #Add title and space out subplots
    fig.suptitle(label+f'\n$S(0)=K={option.X0}, M={M}$')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94],h_pad=2,w_pad=4,pad=4)

##Shows same Brownian path over range of discretisations
def brownian_plot(L=8,M=2):
    """
    Plots Brownian paths over range of dicretisations from Nsteps = M**L to M**0
    
    Parameters:
        L(int) = 8 : indice of finest level
        M(int) = 2 : coarseness factor
    """
    Nsteps=M**L
    dt=1/Nsteps
    dWf=np.random.randn(Nsteps)*np.sqrt(dt) #Brownian motion for adaptive time step
    for l in range(L-1,-1,-1):
        dt=2*dt
        r=np.arange(0,M)
        dWc=np.zeros(M**l)
        for el in r:
            dWc+=dWf[el::M]
        Wc=[0]
        for dw in dWc:
            Wc.append(Wc[-1]+dw)
        tc=np.arange(0,1+dt,dt)
        plt.plot(tc,np.array(Wc)-l,'k-',marker='s',label=f'$l={l}$',markersize=figsize[0])
        dWf=dWc
    plt.legend(loc='upper right',framealpha=1,frameon=True)
    plt.xlabel('$T$')
    plt.title(f'Discrete Approximations of the same Brownian path, $N={M}^l$')