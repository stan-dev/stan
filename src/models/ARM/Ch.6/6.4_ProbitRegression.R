library(rstan)
library(ggplot2)
source("wells.data.R")    

## Probit or logit (wells_logit.stan)
## glm (switch ~ dist100, family=binomial(link="probit"))
if (!file.exists("wells_probit.sm.RData")) {
    rt <- stanc("wells_probit.stan", model_name="wells_probit")
    wells_probit.sm <- stan_model(stanc_ret=rt)
    save(wells_probit.sm, file="wells_probit.sm.RData")
} else {
    load("wells_probit.sm.RData", verbose=TRUE)
}

dataList <- list(N=N, switc=switc, dist=dist)
wells_probit.sf1 <- sampling(wells_probit.sm, dataList)
print(wells_probit.sf1)

 # Figure 6.2
m <- ggplot(data.frame(x = c(0, 2)), aes(x))
m + stat_function(geom="line", fun=dnorm, arg=list(mean=0, sd=1.6)) + scale_x_continuous(limits=c(-6,6)) + theme_bw()
