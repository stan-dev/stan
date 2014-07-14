library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Complete pooling model

radon.pooling.sf <- stan(file='radon.pooling.stan', data=radon.data,
                         iter=1000, chains=4)
print(radon.pooling.sf)

## No pooling model

radon.nopooling.sf <- stan(file='radon.nopooling.stan', data=radon.data,
                           iter=1000, chains=4)
print(radon.nopooling.sf)

## Multilevel model with group-level predictors

radon.2.sf <- stan(file='radon.2.stan', data=radon.data,
                   iter=1000, chains=4)
print(radon.2.sf)
