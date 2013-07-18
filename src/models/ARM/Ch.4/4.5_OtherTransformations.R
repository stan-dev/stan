library(rstan)
library(ggplot2)
library(arm)
source("kid_iq.data.R")    

## Fit the model (kid_iq_factor.stan)
## lm (kid_score ~ as.factor(mom_work))
if (!file.exists("kid_iq_factor.sm.RData")) {
    rt <- stanc("kid_iq_factor.stan", model_name="kid_iq_factor")
    kid_iq_factor.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_factor.sm, file="kid_iq_factor.sm.RData")
} else {
    load("kid_iq_factor.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, kid_score=kid_score, mom_work=mom_work)
kid_iq_factor.sf1 <- sampling(kid_iq_factor.sm, dataList.1)
print(kid_iq_factor.sf1)
