library(rstan)
library(ggplot2)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq
data.list <- c("N", "kid_score", "mom_hs", "mom_iq", "mom_hs_new", "mom_iq_new")
kidiq_prediction.sf <- stan(file='kidiq_prediction.stan', data=data.list,
                            iter=1000, chains=4)
print(kidiq_prediction.sf, pars = c("kid_score_pred"), prob = c(0.025, 0.975))


### Data

source("kids_before1987.data.R", echo = TRUE)

### Model: ppvt ~ hs + afqt
data.list <- c("N", "ppvt", "hs", "afqt")
kidiq_pre1987.sf <- stan(file='kidiq_validation.stan', data=data.list,
                         iter=1000, chains=4)
print(kidiq_pre1987.sf, pars = c("beta", "sigma", "lp__"))

### External validation

## Data

source("kids_after1987.data.R", echo = TRUE)

## Predicted scores

beta.post <- extract(kidiq_pre1987.sf, "beta")$beta
beta.mean <- colMeans(beta.post)
cscores.new <- beta.mean[1] + beta.mean[2] * hs_ev + beta.mean[3] * afqt_ev
resid <- ppvt_ev - cscores.new
resid.sd <- sd(resid)

## Figure 3.13
# left
p1 <- ggplot(data.frame(cscores.new, ppvt_ev), aes(x = cscores.new, y = ppvt_ev)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1) +
    scale_x_continuous("Predicted score", limits = c(20, 140), breaks = seq(20, 140, 20)) +
    scale_y_continuous("Actual score", limits = c(20, 140), breaks = seq(20, 140, 20)) +
    theme_bw()
print(p1)
# right
dev.new()
p2 <- ggplot(data.frame(cscores.new, resid), aes(x = cscores.new, y = resid)) +
    geom_point() +
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = c(-resid.sd, resid.sd), linetype = "dashed") +
    scale_x_continuous("Predicted score", breaks = seq(70, 100, 10)) +
    scale_y_continuous("Prediction error", breaks = seq(-60, 40, 20)) +
    theme_bw()
print(p2)
