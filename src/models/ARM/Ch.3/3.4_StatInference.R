library(rstan)
library(ggplot2)

### Data
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### Model (kid_iq_multi_preds.stan)
### lm (kid_score ~ mom_hs + mom_iq)

if (!file.exists("kid_iq_multi_preds.sm.RData")) {
    rt <- stanc("kid_iq_multi_preds.stan", model_name="kid_iq_multi_preds")
    kid_iq_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_multi_preds.sm, file="kid_iq_multi_preds.sm.RData")
} else {
    load("kid_iq_multi_preds.sm.RData", verbose=TRUE)
}

dataList <- list(N=N, kid_score=kid_score, mom_hs=mom_hs, mom_iq=mom_iq)
kid_iq_multi_preds.sf <- sampling(kid_iq_multi_preds.sm, dataList)
print(kid_iq_multi_preds.sf)
