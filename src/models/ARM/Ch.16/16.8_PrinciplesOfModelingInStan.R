library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Set up and call Stan

radon.2.sf <- stan(file='radon.2.stan', data=radon.data, iter=1000, chains=4)
print(radon.2.sf)

## Specify the values of sigma_y and sigma_a
## in the transformed data block (see radon.3.stan)

radon.3.sf <- stan(file='radon.3.stan', data=radon.data, iter=1000, chains=4)
print(radon.3.sf)
