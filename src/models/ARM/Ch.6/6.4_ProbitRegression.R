library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

## Probit or logit (wells_probit.stan)
## glm (switch ~ dist100, family=binomial(link="probit"))

dataList <- c("N","switc","dist")
wells_probit.sf1 <- stan(file='wells_probit.stan', data=dataList,
                         iter=1000, chains=4)
print(wells_probit.sf1)

 # Figure 6.2
p1 <- ggplot(data.frame(x = c(0, 2)), aes(x)) +
      stat_function(geom="line", fun=dnorm, arg=list(mean=0, sd=1.6)) +
      scale_x_continuous(limits=c(-6,6)) +
      theme_bw()
print(p1)
