library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Set up and call Stan

if (!exists("radon.2.sm")) {
    if (file.exists("radon.2.sm.RData")) {
        load("radon.2.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.2.stan", model_name = "radon.2")
        radon.2.sm <- stan_model(stanc_ret = rt)
        save(radon.2.sm, file = "radon.2.sm.RData")
    }
}
radon.2.sf <- sampling(radon.2.sm, radon.data)
print(radon.2.sf)

## Specify the values of sigma_y and sigma_a
## in the transformed data block (see radon.3.stan)

if (!exists("radon.3.sm")) {
    if (file.exists("radon.3.sm.RData")) {
        load("radon.3.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.3.stan", model_name = "radon.3")
        radon.3.sm <- stan_model(stanc_ret = rt)
        save(radon.3.sm, file = "radon.3.sm.RData")
    }
}
radon.3.sf <- sampling(radon.3.sm, radon.data)
print(radon.3.sf)
