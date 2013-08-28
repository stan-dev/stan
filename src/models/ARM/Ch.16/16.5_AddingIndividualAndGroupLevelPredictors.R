library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Complete pooling model

if (!exists("radon.pooling.sm")) {
    if (file.exists("radon.pooling.sm.RData")) {
        load("radon.pooling.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.pooling.stan", model_name = "radon.pooling")
        radon.pooling.sm <- stan_model(stanc_ret = rt)
        save(radon.pooling.sm, file = "radon.pooling.sm.RData")
    }
}
radon.pooling.sf <- sampling(radon.pooling.sm, radon.data)
print(radon.pooling.sf)

## No pooling model

if (!exists("radon.nopooling.sm")) {
    if (file.exists("radon.nopooling.sm.RData")) {
        load("radon.nopooling.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.nopooling.stan", model_name = "radon.nopooling")
        radon.nopooling.sm <- stan_model(stanc_ret = rt)
        save(radon.nopooling.sm, file = "radon.nopooling.sm.RData")
    }
}
radon.nopooling.sf <- sampling(radon.nopooling.sm, radon.data)
print(radon.nopooling.sf)

## Multilevel model with group-level predictors

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
