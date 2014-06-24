library(rstan)

### Data

source("wells.data.R", echo = TRUE)

### Model: switched ~ dist/100 + arsenic + educ/4

data.list <- c("N", "switched", "dist", "arsenic", "educ")
wells_dae.sf <- stan(file='wells_dae.stan', data=data.list,
                     iter=1000, chains=4)
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

wells_dae_inter.sf <- stan(file='wells_dae_inter.stan', data=data.list,
                           iter=1000, chains=4)
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

