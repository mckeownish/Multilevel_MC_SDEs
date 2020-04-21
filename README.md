# Multilevel_MC_SDEs
RSCAM Group 1 Project (Multilevel Monte Carlo for SDEs)

# Contributors 
 Olena Balan<br/>
 Chantal Kool<br/>
 Isabell Linde<br/>
 Josh McKeown<br/>
 Luke Shaw
 
```
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

Example usage:
import mlmc_RSCAM as mlmc
opt = mlmc.Euro_GBM(X0=125)
sums,N=opt.mlmc(eps=0.01)
print(sum(sums[0,:]/N),opt.BS()) #Compare BS price with mlmc-calculated price
mlmc.variance_plot(opt,eps=0.01,label='European GBM ') #Plot mean/variance plot
opt.asset_plot(L=4,M=4) #Plot asset price on two discretisation levels
```
