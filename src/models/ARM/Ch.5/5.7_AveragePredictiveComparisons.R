library(rstan)
library(ggplot2)
source("wells.data.R")    

## Fitting the model (wells_edu.stan)
##  glm (switch ~ dist100 + arsenic + educ4, family=binomial(link="logit"))
if (!file.exists("wells_edu.sm.RData")) {
    rt <- stanc("wells_edu.stan", model_name="wells_edu")
    wells_edu.sm <- stan_model(stanc_ret=rt)
    save(wells_edu.sm, file="wells_edu.sm.RData")
} else {
    load("wells_edu.sm.RData", verbose=TRUE)
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
delta <- invlogit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ/4) -
         invlogit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ/4)
print (mean(delta))

 #  for arsenic level

hi <- 1.0
lo <- 0.5
delta <- invlogit (b[1] + b[2]*dist/100 + b[3]*hi + b[4]*educ/4) -
         invlogit (b[1] + b[2]*dist/100 + b[3]*lo + b[4]*educ/4)
print (mean(delta))

 # for education

hi <- 3
lo <- 0
delta <- invlogit (b[1]+b[2]*dist/100+b[3]*arsenic+b[4]*hi) -
         invlogit (b[1]+b[2]*dist/100+b[3]*arsenic+b[4]*lo)
print (mean(delta))

## Avg predictive comparisons with interactions (wells_all.stan)
##  glm (switch ~ dist100 + arsenic + educ4 + dist100:arsenic, family=binomial(link="logit"))
if (!file.exists("wells_all.sm.RData")) {
    rt <- stanc("wells_all.stan", model_name="wells_all")
    wells_all.sm <- stan_model(stanc_ret=rt)
    save(wells_all.sm, file="wells_all.sm.RData")
} else {
    load("wells_all.sm.RData", verbose=TRUE)
}
dataList.2 <- list(N=N, switc=switc, dist=dist,arsenic=arsenic,educ=educ)
wells_all.sf1 <- sampling(wells_all.sm, dataList.2)
print(wells_all.sf1)

 # for distance
beta.post <- extract(wells_all.sf1, "beta")$beta
b <- colMeans(beta.post)
hi <- 1
lo <- 0
delta <- invlogit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ/4 +
                   b[5]*hi*arsenic) -
         invlogit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ/4 +
                   b[5]*lo*arsenic)
print (mean(delta))


