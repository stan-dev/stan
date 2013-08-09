library(rstan)

### Data

source("wells.data.R", echo = TRUE)

### Model: switched ~ dist/100 + arsenic + educ/4

if (!exists("wells_dae.sm")) {
    if (file.exists("wells_dae.sm.RData")) {
        load("wells_dae.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dae.stan", model_name = "wells_dae")
        wells_dae.sm <- stan_model(stanc_ret = rt)
        save(wells_dae.sm, file = "wells_dae.sm.RData")
    }
}

data.list <- c("N", "switched", "dist", "arsenic", "educ")
wells_dae.sf <- sampling(wells_dae.sm, data.list)
print(wells_dae.sf, pars = c("beta", "lp__"))

## Avg predictive differences

b <- colMeans(extract(wells_dae.sf, "beta")$beta)
invlogit <- function(x) plogis(x)

# for distance to nearest safe well

hi <- 1
lo <- 0
delta <- invlogit(b[1] + b[2] * hi + b[3] * arsenic + b[4] * educ / 4) -
         invlogit(b[1] + b[2] * lo + b[3] * arsenic + b[4] * educ / 4)
print(mean(delta))

# for arsenic level

hi <- 1.0
lo <- 0.5
delta <- invlogit(b[1] + b[2] * dist / 100 + b[3] * hi + b[4] * educ / 4) -
         invlogit(b[1] + b[2] * dist / 100 + b[3] * lo + b[4] * educ / 4)
print(mean(delta))

# for education

hi <- 3
lo <- 0
delta <- invlogit(b[1] + b[2] * dist / 100 + b[3] * arsenic + b[4] * hi) -
         invlogit(b[1] + b[2] * dist / 100 + b[3] * arsenic + b[4] * lo)
print(mean(delta))

### Avg predictive comparisons with interactions
##  switches ~ dist/100 + arsenic + educ/4 + dist/100:arsenic

if (!exists("wells_dae_inter.sm")) {
    if (file.exists("wells_dae_inter.sm.RData")) {
        load("wells_dae_inter.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dae_inter.stan", model_name = "wells_dae_inter")
        wells_dae_inter.sm <- stan_model(stanc_ret = rt)
        save(wells_dae_inter.sm, file = "wells_dae_inter.sm.RData")
    }
}

wells_dae_inter.sf <- sampling(wells_dae_inter.sm, data.list)
print(wells_dae_inter.sf)

b <- colMeans(extract(wells_dae_inter.sf, "beta")$beta)

# for distance

hi <- 1
lo <- 0
delta <- invlogit(b[1] + b[2] * hi + b[3] * arsenic + b[4] * educ / 4 +
                  b[5] * hi * arsenic) -
         invlogit(b[1] + b[2] * lo + b[3] * arsenic + b[4] * educ / 4 +
                  b[5] * lo * arsenic)
print(mean(delta))

