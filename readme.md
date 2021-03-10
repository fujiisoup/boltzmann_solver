# A program and data used for the article at arXiv:2103.04463

## Program
### A simple solver of isotropic 0d-Boltzman's equation

Two models are implemented.

+ Linear model. `core.BoltzmannLinear`  
The test particles collide only with the thermal particles with a different mass. This particles are heated to some very high energy with given rate.
+ Mixture model. `core.BoltzmannMixture`  
The test particles collide both with the different- and same-kind particles. 

## Data
+ Balmer-$\alpha$ spectrum observed for the commercial spectral lamp  
`experimental_data/spectral_lamp/`
+ Extracted data from Amorim et al  
`experimental_data/amorim/`
+ Data from NSO historical data archive  
`experimental_data/NSO/`