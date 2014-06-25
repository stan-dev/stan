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

dataList.1 <- list(N=length(watched), watched=watched,encouraged=encouraged)
sesame_one_pred_a.sf1 <- stan(file='sesame_one_pred_a.stan', data=dataList.1,
                              iter=1000, chains=4)
print(sesame_one_pred_a.sf1)

beta.post <- extract(sesame_one_pred_a.sf1, "beta")$beta
beta.mean1 <- colMeans(beta.post)

## (sesame_one_pred_b.stan)
## lm (y ~ encouraged)

dataList.2 <- list(N=length(y), watched=y,encouraged=encouraged)
sesame_one_pred_b.sf1 <- stan(file='sesame_one_pred_a.stan', data=dataList.2,
                              iter=1000, chains=4)
print(sesame_one_pred_b.sf1)

beta.post <- extract(sesame_one_pred_b.sf1, "beta")$beta
beta.mean2 <- colMeans(beta.post)


iv.est.1 <- beta.mean2[2] / beta.mean1[2]
print(iv.est.1)
