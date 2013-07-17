stopifnot(require(rstan))
library(ggplot2)
source("wells.data.R")    

## Histogram on distance (Figure 5.8)
frame = data.frame(dist=dist)
m <- ggplot(frame,aes(x=dist))  + scale_x_continuous("Distance (in meters) to the nearest safe well")
m + geom_histogram(colour = "black", fill = "white", binwidth=10) + theme_bw()

## Logistic regression with one predictor (wells_one_pred.stan)
if (!file.exists("wells_one_pred.sm.RData")) {
    rt <- stanc("wells_one_pred.stan", model_name="wells_one_pred")
    wells_one_pred.sm <- stan_model(stanc_ret=rt)
    save(wells_one_pred.sm, file="wells_one_pred.sm.RData")
} else {
    load("wells_one_pred.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, switc=switc, dist=dist)
wells_one_pred.sf1 <- sampling(wells_one_pred.sm, dataList.1)
print(wells_one_pred.sf1)

beta.post <- extract(wells_one_pred.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)
## Repeat the regression above with distance in 100-meter units (wells_one_pred_scale.stan)
dist100 <- dist/100
dataList.2 <- list(N=N, switc=switc, dist=dist100)
wells_one_pred.sf2 <- sampling(wells_one_pred.sm, dataList.2)
print(wells_one_pred.sf2)

## Graphing the fitted model with one predictor (Figure 5.9)
jitter.binary <- function(a, jitt=.05){
  ifelse (a==0, runif (length(a), 0, jitt), runif (length(a), 1-jitt, 1))
}

switch.jitter <- jitter.binary(switc)
frame1 = data.frame(dist=dist,switc=switch.jitter)
m2 <- ggplot(frame1,aes(x=dist,y=switc))
m2 <- m2 + geom_point() + scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) + scale_x_continuous("Distance (in meters) to nearest safe well") + theme_bw() + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean[1] - beta.mean[2] * x)))

## Histogram on arsenic levels (Figure 5.10)
frame3 = data.frame(ars=arsenic)
m3 <- ggplot(frame3,aes(x=ars))  + scale_x_continuous("Arsenic concentration in well water")
m3 + geom_histogram(colour = "black", fill = "white", binwidth=0.25) + theme_bw()

## Logistic regression with second input variable (wells_two_pred.stan)
if (!file.exists("wells_two_pred.sm.RData")) {
    rt <- stanc("wells_two_pred.stan", model_name="wells_two_pred")
    wells_two_pred.sm <- stan_model(stanc_ret=rt)
    save(wells_two_pred.sm, file="wells_two_pred.sm.RData")
} else {
    load("wells_two_pred.sm.RData", verbose=TRUE)
}
dataList.3 <- list(N=N, switc=switc, dist=dist,arsenic=arsenic)
wells_two_pred.sf1 <- sampling(wells_two_pred.sm, dataList.3)
print(wells_two_pred.sf1)

beta.post2 <- extract(wells_two_pred.sf1, "beta")$beta
beta.mean2 <- colMeans(beta.post2)

## Graphing the fitted model with two predictors (Figure 5.11)
frame3 = data.frame(dist=dist,switc=switch.jitter)
m2 <- ggplot(frame3,aes(x=dist,y=switc))
m2 <- m2 + geom_point() + scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) + scale_x_continuous("Distance (in meters) to nearest safe well") + theme_bw() + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean2[1] - 0.5 * beta.mean2[3] - beta.mean[2] * x / 100.0))) + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean2[1] - beta.mean2[3] - beta.mean[2] * x / 100.0)))

frame4 = data.frame(ars=arsenic,switc=switch.jitter)
m2 <- ggplot(frame4,aes(x=dist,y=switc))
m2 <- m2 + geom_point() + scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) + scale_x_continuous("Arsenic concentration in well water") + theme_bw() + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean2[1] - beta.mean2[3] * x))) + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean2[1] - beta.mean2[2] * 0.5 - beta.mean2[3] * x)))
