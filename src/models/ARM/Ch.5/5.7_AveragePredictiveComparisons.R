library(rstan)
library(ggplot2)
library(boot)
source("wells.data.R", echo = TRUE)    

## Fitting the model (wells_edu.stan)
##  glm (switch ~ dist100 + arsenic + educ4, family=binomial(link="logit"))
if (!exists("wells_edu.sm")) {
    if (file.exists("wells_edu.sm.RData")) {
        load("wells_edu.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_edu.stan", model_name = "wells_edu")
        wells_edu.sm <- stan_model(stanc_ret = rt)
        save(wells_edu.sm, file = "wells_edu.sm.RData")
    }
}
dataList.1 <- list(N=N, switc=switc, dist=dist,arsenic=arsenic,assoc=assoc,educ=educ)
wells_edu.sf1 <- sampling(wells_edu.sm, dataList.1)
print(wells_edu.sf1)

## Avg predictive differences

beta.post <- extract(wells_edu.sf1, "beta")$beta
b <- colMeans(beta.post)

 # for distance to nearest safe well

hi <- 1
lo <- 0
delta <- inv.logit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ/4) -
         inv.logit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ/4)
print (mean(delta))

 #  for arsenic level

hi <- 1.0
lo <- 0.5
delta <- inv.logit (b[1] + b[2]*dist/100 + b[3]*hi + b[4]*educ/4) -
         inv.logit (b[1] + b[2]*dist/100 + b[3]*lo + b[4]*educ/4)
print (mean(delta))

 # for education

hi <- 3
lo <- 0
delta <- inv.logit (b[1]+b[2]*dist/100+b[3]*arsenic+b[4]*hi) -
         inv.logit (b[1]+b[2]*dist/100+b[3]*arsenic+b[4]*lo)
print (mean(delta))

## Avg predictive comparisons with interactions (wells_all.stan)
##  glm (switch ~ dist100 + arsenic + educ4 + dist100:arsenic, family=binomial(link="logit"))
if (!exists("wells_all.sm")) {
    if (file.exists("wells_all.sm.RData")) {
        load("wells_all.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_all.stan", model_name = "wells_all")
        wells_all.sm <- stan_model(stanc_ret = rt)
        save(wells_all.sm, file = "wells_all.sm.RData")
    }
}
dataList.2 <- c("N","switc","dist","arsenic","educ")
wells_all.sf1 <- sampling(wells_all.sm, dataList.2)
print(wells_all.sf1)

 # for distance
beta.post <- extract(wells_all.sf1, "beta")$beta
b <- colMeans(beta.post)
hi <- 1
lo <- 0
delta <- inv.logit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ/4 +
                   b[5]*hi*arsenic) -
         inv.logit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ/4 +
                   b[5]*lo*arsenic)
print (mean(delta))


