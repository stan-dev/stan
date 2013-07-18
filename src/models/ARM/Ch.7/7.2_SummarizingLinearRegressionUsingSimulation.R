library(rstan)
library(ggplot2)
library(arm)
source("earnings.data.R")

## Simulation to represent predictive uncertainty
 # Model of log earnings with interactions (earnings_interactions.stan)
## lm(log.earn ~ height + male + height:male)
if (!file.exists("earnings_interactions.sm.RData")) {
    rt <- stanc("earnings_interactions.stan", model_name="earnings_interactions")
    earnings_interactions.sm <- stan_model(stanc_ret=rt)
    save(earnings_interactions.sm, file="earnings_interactions.sm.RData")
} else {
    load("earnings_interactions.sm.RData", verbose=TRUE)
}
dataList.3 <- list(N=N, earnings=earnings, height=height,sex=sex)
earnings_interactions.sf1 <- sampling(earnings_interactions.sm, dataList.3)
print(earnings_interactions.sf1)

 # Prediction
log.earn <- log(earnings)
male <- 2 - sex
earn.logmodel.3 <- lm (log.earn ~ height + male + height:male)
x.new <- data.frame (height=68, male=1)
pred.interval <- predict (earn.logmodel.3, x.new, interval="prediction", 
  level=.95)

print (exp (pred.interval))

## Constructing the predictive interval using simulation

pred <- exp (rnorm (1000, 9.95, .88))
pred.original.scale <- rnorm (1000, 9.95, .88)

 # Histograms (Figure 7.2)
frame1 = data.frame(x1=pred.original.scale)
m <- ggplot(frame1,aes(x=x1))  + scale_x_continuous("log(earnings)")
m + geom_histogram(colour = "black", fill = "white") + theme_bw()

frame2 = data.frame(x1=pred)
m2 <- ggplot(frame1,aes(x=x1))  + scale_x_continuous("earnings")
m2 + geom_histogram(colour = "black", fill = "white") + theme_bw()

## Why do we need simulation for predictive inferences?

pred.man <- exp (rnorm (1000, 8.4 + 0.17*68 - 0.079*1 + .007*68*1, .88))
pred.woman <- exp (rnorm (1000, 8.4 + 0.17*68 - 0.079*0 + .007*68*0, .88))
pred.diff <- pred.man - pred.woman
pred.ratio <- pred.man/pred.woman

## Simulation to represent uncertainty in regression coefficients
n.sims <- 1000
fit.1<- lm (log.earn ~ height + male + height:male)
sim.1 <- sim (fit.1, n.sims)

height.coef <- sim.1$coef[,2]
mean (height.coef)
sd (height.coef)
quantile (height.coef, c(.025, .975))

height.for.men.coef <- sim.1$coef[,2] + sim.1$coef[,4]
quantile (height.for.men.coef, c(.025, .975))
