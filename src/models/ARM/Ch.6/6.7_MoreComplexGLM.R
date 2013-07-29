library(rstan)
library(ggplot2)
source("earnings1.data.R", echo = TRUE)    

## Mixed discrete/continuous data

# Logistic regression with interactions (earnings1.stan)
# glm (earn.pos ~ height + male, family=binomial(link="logit"))
if (!exists("earnings1.sm")) {
    if (file.exists("earnings1.sm.RData")) {
        load("earnings1.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("earnings1.stan", model_name = "earnings1")
        earnings1.sm <- stan_model(stanc_ret = rt)
        save(earnings1.sm, file = "earnings1.sm.RData")
    }
}
dataList.1 <- c("N","earn_pos","height","male")
earnings1.sf1 <- sampling(earnings1.sm, dataList.1)
print(earnings1.sf1)

source("earnings2.data.R", echo = TRUE)    
# Logistic regression with interactions (earnings2.stan)
# lm (log.earn ~ height + male, subset=earn>0)
if (!exists("earnings2.sm")) {
    if (file.exists("earnings2.sm.RData")) {
        load("earnings2.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("earnings2.stan", model_name = "earnings2")
        earnings2.sm <- stan_model(stanc_ret = rt)
        save(earnings2.sm, file = "earnings2.sm.RData")
    }
}
dataList.2 <- c("N","earnings","height","male")
earnings2.sf1 <- sampling(earnings2.sm, dataList.2)
print(earnings2.sf1)
