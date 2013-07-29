library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model (kidiq_multi_preds.stan): kid_score ~ mom_hs + mom_iq

if (!exists("kidiq_multi_preds.sm")) {
    if (file.exists("kidiq_multi_preds.sm.RData")) {
        load("kidiq_multi_preds.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_multi_preds.stan", model_name = "kidiq_multi_preds")
        kidiq_multi_preds.sm <- stan_model(stanc_ret = rt)
        save(kidiq_multi_preds.sm, file = "kidiq_multi_preds.sm.RData")
    }
}

data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_multi_preds.sf <- sampling(kidiq_multi_preds.sm, data.list)
print(kidiq_multi_preds.sf, pars = c("beta", "sigma", "lp__"))
