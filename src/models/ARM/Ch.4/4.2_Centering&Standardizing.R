library(rstan)
library(ggplot2)

### Data
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### Model (kid_iq_interaction.stan)
## lm (kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq)
if (!file.exists("kid_iq_interaction.sm.RData")) {
    rt <- stanc("kid_iq_interaction.stan", model_name="kid_iq_interaction")
    kid_iq_interaction.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_interaction.sm, file="kid_iq_interaction.sm.RData")
} else {
    load("kid_iq_interaction.sm.RData", verbose=TRUE)
}

dataList <- list(N=N, kid_score=kid_score, mom_hs=mom_hs, mom_iq=mom_iq)
kid_iq_interaction.sf <- sampling(kid_iq_interaction.sm, dataList)
print(kid_iq_interaction.sf)

# centering by subtracting the mean (kid_iq_center_mean.stan)
# lm (kid_score ~ c_mom_hs + c_mom_iq + c_mom_hs:c_mom_iq)
c_mom_hs <- mom_hs - mean(mom_hs)
c_mom_iq <- mom_iq - mean(mom_iq)

dataList2 <- list(N=N, kid_score=kid_score, mom_hs=c_mom_hs, mom_iq=c_mom_iq)
kid_iq_interaction2.sf <- sampling(kid_iq_interaction.sm, dataList2)
print(kid_iq_interaction2.sf)

# using a conventional centering point (kid_iq_center_conventional.stan)
# lm (kid_score ~ c2_mom_hs + c2_mom_iq + c2_mom_hs:c2_mom_iq)
c2_mom_hs <- mom_hs - 0.5
c2_mom_iq <- mom_iq - 100

dataList3 <- list(N=N, kid_score=kid_score, mom_hs=c2_mom_hs, mom_iq=c2_mom_iq)
kid_iq_interaction3.sf <- sampling(kid_iq_interaction.sm, dataList3)
print(kid_iq_interaction3.sf)

# centering by subtracting the mean & dividing by 2 sd (kid_iq_center_z.stan)
# lm (kid_score ~ z_mom_hs + z_mom_iq + z_mom_hs:z_mom_iq)
z_mom_hs <- (mom_hs - mean(mom_hs))/(2*sd(mom_hs))
z_mom_iq <- (mom_iq - mean(mom_iq))/(2*sd(mom_iq))

dataList4 <- list(N=N, kid_score=kid_score, mom_hs=z_mom_hs, mom_iq=z_mom_iq)
kid_iq_interaction4.sf <- sampling(kid_iq_interaction.sm, dataList4)
print(kid_iq_interaction4.sf)
