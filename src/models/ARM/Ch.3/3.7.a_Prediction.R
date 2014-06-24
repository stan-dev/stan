library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq
data.list <- c("N", "kid_score", "mom_hs", "mom_iq", "mom_hs_new", "mom_iq_new")
kidiq_prediction.sf <- stan(file='kidiq_prediction.stan', data=data.list,
                            iter=1000, chains=4)
print(kidiq_prediction.sf, pars = c("kid_score_pred"), prob = c(0.025, 0.975))
