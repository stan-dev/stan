library(rstan)
library(ggplot2)
library(foreign)

sesame <- read.dta("sesame.dta")
attach(sesame)

## Rename variables of interest
watched <- regular
encouraged <- encour
y <- postlet

## Instrumental variables estimate (sesame_one_pred_a.stan)
## lm (watched ~ encouraged)
if (!exists("sesame_one_pred_a.sm")) {
    if (file.exists("sesame_one_pred_a.sm.RData")) {
        load("sesame_one_pred_a.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("sesame_one_pred_a.stan", model_name = "sesame_one_pred_a")
        sesame_one_pred_a.sm <- stan_model(stanc_ret = rt)
        save(sesame_one_pred_a.sm, file = "sesame_one_pred_a.sm.RData")
    }
}

dataList.1 <- list(N=length(watched), watched=watched,encouraged=encouraged)
sesame_one_pred_a.sf1 <- sampling(sesame_one_pred_a.sm, dataList.1)
print(sesame_one_pred_a.sf1)

beta.post <- extract(sesame_one_pred_a.sf1, "beta")$beta
beta.mean1 <- colMeans(beta.post)

## (sesame_one_pred_b.stan)
## lm (y ~ encouraged)

dataList.2 <- list(N=length(y), watched=y,encouraged=encouraged)
sesame_one_pred_b.sf1 <- sampling(sesame_one_pred_a.sm, dataList.2)
print(sesame_one_pred_b.sf1)

beta.post <- extract(sesame_one_pred_b.sf1, "beta")$beta
beta.mean2 <- colMeans(beta.post)


iv.est.1 <- beta.mean2[2] / beta.mean1[2]
print(iv.est.1)
