library(rstan)
library(ggplot2)
source("wells.data.R")    

## Fitting the model (wells_logit.stan)
if (!file.exists("wells_logit.sm.RData")) {
    rt <- stanc("wells_logit.stan", model_name="wells_logit")
    wells_logit.sm <- stan_model(stanc_ret=rt)
    save(wells_logit.sm, file="wells_logit.sm.RData")
} else {
    load("wells_logit.sm.RData", verbose=TRUE)
}

## Probit or logit
## glm (switch ~ dist100, family=binomial(link="logit"))
dataList <- list(N=N, switc=switc, dist=dist)
wells_logit.sf1 <- sampling(wells_logit.sm, dataList)
print(wells_logit.sf1)

