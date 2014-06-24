library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model (kidiq_multi_preds.stan): kid_score ~ mom_hs + mom_iq

data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_multi_preds <- stan(file='kidiq_multi_preds.stan', data=data.list,
                          iter=1000, chains=4)
print(kidiq_multi_preds, pars = c("beta", "sigma", "lp__"))
