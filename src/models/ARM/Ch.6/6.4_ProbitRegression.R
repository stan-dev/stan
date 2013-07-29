library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

## Probit or logit (wells_probit.stan)
## glm (switch ~ dist100, family=binomial(link="probit"))
if (!exists("wells_probit.sm")) {
    if (file.exists("wells_probit.sm.RData")) {
        load("wells_probit.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_probit.stan", model_name = "wells_probit")
        wells_probit.sm <- stan_model(stanc_ret = rt)
        save(wells_probit.sm, file = "wells_probit.sm.RData")
    }
}

dataList <- c("N","switc","dist")
wells_probit.sf1 <- sampling(wells_probit.sm, dataList)
print(wells_probit.sf1)

 # Figure 6.2
p1 <- ggplot(data.frame(x = c(0, 2)), aes(x)) +
      stat_function(geom="line", fun=dnorm, arg=list(mean=0, sd=1.6)) +
      scale_x_continuous(limits=c(-6,6)) +
      theme_bw()
print(p1)
