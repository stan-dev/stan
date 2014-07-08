library(rstan)
library(ggplot2)
source("earnings.data.R", echo = TRUE)

## Simulation to represent predictive uncertainty
 # Model of log earnings with interactions (earnings_interactions.stan)
## lm(log.earn ~ height + male + height:male)

dataList.3 <- list(N=N, earnings=earnings, height=height,sex=sex1)
earnings_interactions.sf1 <- stan(file='earnings_interactions.stan',
                                  data=dataList.3,
                                  iter=1000, chains=4)
print(earnings_interactions.sf1)
post <- extract(earnings_interactions.sf1)

 # Prediction
log.earn <- log(earnings)
male <- 2 - sex1
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
p1 <- ggplot(frame1,aes(x=x1))  +
      scale_x_continuous("log(earnings)") +
      geom_histogram(colour = "black", fill = "white") +
      theme_bw()
print(p1)

dev.new()
frame2 = data.frame(x1=pred)
p2 <- ggplot(frame1,aes(x=x1))  +
      scale_x_continuous("earnings") +
      geom_histogram(colour = "black", fill = "white") +
      theme_bw()
print(p2)

## Why do we need simulation for predictive inferences?

pred.man <- exp (rnorm (1000, 8.4 + 0.17*68 - 0.079*1 + .007*68*1, .88))
pred.woman <- exp (rnorm (1000, 8.4 + 0.17*68 - 0.079*0 + .007*68*0, .88))
pred.diff <- pred.man - pred.woman
pred.ratio <- pred.man/pred.woman

## Simulation to represent uncertainty in regression coefficients
n.sims <- 1000

height.coef <- post$beta[,2]
mean (height.coef)
sd (height.coef)
quantile (height.coef, c(.025, .975))

height.for.men.coef <- post$beta[,2] + post$beta[,4]
quantile (height.for.men.coef, c(.025, .975))
