library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq

if (!exists("kidiq_prediction.sm")) {
    if (file.exists("kidiq_prediction.sm.RData")) {
        load("kidiq_prediction.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_prediction.stan", model_name = "kidiq_prediction")
        kidiq_prediction.sm <- stan_model(stanc_ret = rt)
        save(kidiq_prediction.sm, file = "kidiq_prediction.sm.RData")
    }
}

data.list <- c("N", "kid_score", "mom_hs", "mom_iq", "mom_hs_new", "mom_iq_new")
kidiq_prediction.sf <- sampling(kidiq_prediction.sm, data.list)
print(kidiq_prediction.sf, pars = c("kid_score_pred"), prob = c(0.025, 0.975))
