library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

## Fitting the model (wells_logit.stan)

## Probit or logit
## glm (switch ~ dist100, family=binomial(link="logit"))
dataList <- c("N","switc","dist")
wells_logit.sf1 <- stan(file='wells_logit.stan', data=dataList,
                        iter=1000, chains=4)
print(wells_logit.sf1)

