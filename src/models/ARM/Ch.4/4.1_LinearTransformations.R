stopifnot(require(rstan))
library(ggplot2)
library(arm)
source("earnings.data.R")    

### First model: earnings ~ height

if (!file.exists("earnings_one_pred.sm.RData")) {
    rt <- stanc("earnings_one_pred.stan", model_name="earnings_one_pred")
    earnings_one_pred.sm <- stan_model(stanc_ret=rt)
    save(earnings_one_pred.sm, file="earnings_one_pred.sm.RData")
} else {
    load("earnings_one_pred.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, earnings=earnings, height=height)
earnings_one_pred.sf1 <- sampling(earnings_one_pred.sm, dataList.1)
print(earnings_one_pred.sf1)

lm.earn <- lm (earnings ~ height)
sim.earn <- sim (lm.earn)

beta.post <- extract(earnings_one_pred.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

## Figure 4.1 (left)
height.jitter.add <- runif (N, -.2, .2)

frame1 = data.frame(height=height+height.jitter.add,earn=earnings)
m <- ggplot(frame1,aes(x=height,y=earn))
m <- m + geom_point() + scale_y_continuous("Earnings") + scale_x_continuous("Height") + theme_bw()
for (i in 1:20)
  m <- m + geom_abline(intercept=coef(sim.earn)[i,1],slope=coef(sim.earn)[i,2],colour="grey")
m + geom_abline(intercept=beta.mean[1],slope=beta.mean[2],colour="red")

## Figure 4.1 (right) 
mm <- ggplot(frame1,aes(x=height,y=earn))
mm <- mm + geom_point() + scale_y_continuous("Earnings",limits=c(-200000,200000)) + scale_x_continuous("Height",limits=c(0,80)) + theme_bw()
for (i in 1:20)
  mm <- mm + geom_abline(intercept=coef(sim.earn)[i,1],slope=coef(sim.earn)[i,2],colour="grey")
mm + geom_abline(intercept=beta.mean[1],slope=beta.mean[2],colour="red")
