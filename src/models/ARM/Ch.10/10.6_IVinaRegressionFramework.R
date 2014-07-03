library(rstan)
library(ggplot2)

source("10.5_CasualEffectsUsingIV.R") # where data was cleaned

## Rename variables of interest

pretest <- prelet

## 2 stage least squares (sesame_one_pred_a.stan)
## lm (watched ~ encouraged)

dataList.1 <- list(N=length(watched), watched=watched,encouraged=encouraged)
sesame_one_pred_2a.sf1 <- stan(file='sesame_one_pred_a.stan', data=dataList.1,
                               iter=1000, chains=4)
print(sesame_one_pred_2a.sf1)
beta.post <- extract(sesame_one_pred_2a.sf1, "beta")$beta
beta.mean2a <- colMeans(beta.post)

watched.hat <- beta.mean2a[1] + beta.mean2a[2] * encouraged

## (sesame_one_pred_2b.stan)
## lm (y ~ watched.hat)

dataList.2 <- list(N=length(y), watched=y,encouraged=watched.hat)
sesame_one_pred_2b.sf1 <- stan(file='sesame_one_pred_a.stan', data=dataList.2,
                               iter=1000, chains=4)
print(sesame_one_pred_2b.sf1)

## Adjusting for covariates in a IV framework (sesame_multi_preds_3a.stan)
## lm (watched ~ encouraged + pretest + as.factor(site) + setting)

dataList.3 <- list(N=length(watched), watched=watched,encouraged=encouraged,pretest=pretest, site=site,setting=setting)
sesame_multi_pred_3a.sf1 <- stan(file='sesame_multi_preds_3a.stan',
                                 data=dataList.3,
                                 iter=1000, chains=4)
print(sesame_multi_pred_3a.sf1)

beta.post <- extract(sesame_multi_pred_3a.sf1, "beta")$beta
beta.mean3a <- colMeans(beta.post)

watched.hat <- beta.mean3a[1] + beta.mean3a[2] * encouraged + beta.mean3a[3] * pretest + beta.mean3a[4] * (site==2) + beta.mean3a[5] * (site==3) + beta.mean3a[6] * (site==4) + beta.mean3a[7] * (site==5) + beta.mean3a[8] * setting

## (sesame_multi_preds_3b.stan)
## lm (y ~ watched.hat + pretest + as.factor(site) + setting)
dataList.4 <- list(N=length(watched.hat), watched=y,encouraged=watched.hat,pretest=pretest, site=site,setting=setting)
sesame_multi_pred_3b.sf1 <- stan(file='sesame_multi_preds_3b.stan',
                                 data=dataList.4,
                                 iter=1000, chains=4)
print(sesame_multi_pred_3b.sf1)

## Se for IV estimates (FIXME)

## Performing 2sls automatically

 # regression without pre-treatment variables

 # regression controlling for pre-treatment variables
