library(rstan)
library(ggplot2)
source("earnings1.data.R", echo = TRUE)    

## Mixed discrete/continuous data

# Logistic regression with interactions (earnings1.stan)
# glm (earn.pos ~ height + male, family=binomial(link="logit"))

dataList.1 <- c("N","earn_pos","height","male")
earnings1.sf1 <- stan(file='earnings1.stan', data=dataList.1,
                      iter=1000, chains=4)
print(earnings1.sf1)

source("earnings2.data.R", echo = TRUE)    
# Logistic regression with interactions (earnings2.stan)
# lm (log.earn ~ height + male, subset=earn>0)
dataList.2 <- c("N","earnings","height","sex")
earnings2.sf1 <- stan(file='earnings2.stan', data=dataList.2,
                      iter=1000, chains=4)
print(earnings2.sf1)
