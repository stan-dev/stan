stopifnot(require(rstan))
library(ggplot2)
source("wells.data.R")    

## Fitting the model (wells_edu.stan)
if (!file.exists("wells_edu.sm.RData")) {
    rt <- stanc("wells_edu.stan", model_name="wells_edu")
    wells_edu.sm <- stan_model(stanc_ret=rt)
    save(wells_edu.sm, file="wells_edu.sm.RData")
} else {
    load("wells_edu.sm.RData", verbose=TRUE)
}
dataList.3 <- list(N=N, switc=switc, dist=dist,arsenic=arsenic,assoc=assoc,educ=educ)
wells_edu.sf1 <- sampling(wells_edu.sm, dataList.3)
print(wells_edu.sf1)

## Avg predictive differences

beta.post <- extract(wells_edu.sf1, "beta")$beta
b <- colMeans(beta.post)

 # for distance to nearest safe well

hi <- 1
lo <- 0
delta <- invlogit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ4) -
         invlogit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ4)
print (mean(delta))

 #  for arsenic level

hi <- 1.0
lo <- 0.5
delta <- invlogit (b[1] + b[2]*dist100 + b[3]*hi + b[4]*educ4) -
         invlogit (b[1] + b[2]*dist100 + b[3]*lo + b[4]*educ4)
print (mean(delta))

 # for education

hi <- 3
lo <- 0
delta <- invlogit (b[1]+b[2]*dist100+b[3]*arsenic+b[4]*hi) -
         invlogit (b[1]+b[2]*dist100+b[3]*arsenic+b[4]*lo)
print (mean(delta))

## Avg predictive comparisons with interactions

fit.11 <- glm (switch ~ dist100 + arsenic + educ4 + dist100:arsenic,
  family=binomial(link="logit"))
display (fit.11)

 # for distance

b <- coef (fit.11)
hi <- 1
lo <- 0
delta <- invlogit (b[1] + b[2]*hi + b[3]*arsenic + b[4]*educ4 +
                   b[5]*hi*arsenic) -
         invlogit (b[1] + b[2]*lo + b[3]*arsenic + b[4]*educ4 +
                   b[5]*lo*arsenic)
print (mean(delta))


