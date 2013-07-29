library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

## Fitting the model (wells_logit.stan)
if (!exists("wells_logit.sm")) {
    if (file.exists("wells_logit.sm.RData")) {
        load("wells_logit.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_logit.stan", model_name = "wells_logit")
        wells_logit.sm <- stan_model(stanc_ret = rt)
        save(wells_logit.sm, file = "wells_logit.sm.RData")
    }
}

## Probit or logit
## glm (switch ~ dist100, family=binomial(link="logit"))
dataList <- c("N","switc","dist")
wells_logit.sf1 <- sampling(wells_logit.sm, dataList)
print(wells_logit.sf1)

