stopifnot(require(rstan))
library(ggplot2)
library(arm)
source("earnings.data.R")    

## Log transformation (earnings_log.stan)
## lm(log.earn ~ height)
if (!file.exists("earnings.sm.RData")) {
    rt <- stanc("earnings.stan", model_name="earnings")
    earnings.sm <- stan_model(stanc_ret=rt)
    save(earnings.sm, file="earnings.sm.RData")
} else {
    load("earnings.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, earnings=log(earnings), height=height)
earnings_log.sf1 <- sampling(earnings.sm, dataList.1)
print(earnings_log.sf1)

earn.logmodel.1 <- lm(log(earnings) ~ height)
sim.logmodel.1 <- sim (earn.logmodel.1)

beta.post <- extract(earnings_log.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

## Figure 4.3
frame1 = data.frame(height=height+ runif(N,-.2,.2),earn=log(earnings))
m <- ggplot(frame1,aes(x=height,y=earn))
m <- m + geom_point() + scale_y_continuous("Log(Earnings)") + scale_x_continuous("Height") + theme_bw()
for (i in 1:20)
  m <- m + geom_abline(intercept=coef(sim.logmodel.1)[i,1],slope=coef(sim.logmodel.1)[i,2],colour="grey")
m + geom_abline(intercept=beta.mean[1],slope=beta.mean[2],colour="red")

frame2 = data.frame(height=height+ runif(N,-.2,.2),earn=(earnings))
mm <- ggplot(frame2,aes(x=height,y=earn))
mm <- mm + geom_point() + scale_y_continuous("Earnings",limits=c(-1000,200000)) + scale_x_continuous("Height") + theme_bw()
for (i in 1:20)
  mm <- mm + geom_abline(intercept=coef(sim.logmodel.1)[i,1],slope=coef(sim.logmodel.1)[i,2],colour="grey",size=2)
mm + geom_abline(intercept=beta.mean[1],slope=beta.mean[2],colour="red")

## Log-base-10 transformation (earnings_log10.stan)
## lm(log10.earn ~ height)
log10.earn <- log10(earnings)

dataList.2 <- list(N=N, earnings=log10.earn, height=height)
earnings_log10.sf1 <- sampling(earnings.sm, dataList.2)
print(earnings_log10.sf1)

## Log scale regression model (earnings_multi_preds.stan)
## lm(log.earn ~ height + male)
if (!file.exists("earnings_multi_preds.sm.RData")) {
    rt <- stanc("earnings_multi_preds.stan", model_name="earnings_multi_preds")
    earnings_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(earnings_multi_preds.sm, file="earnings_multi_preds.sm.RData")
} else {
    load("earnings_multi_preds.sm.RData", verbose=TRUE)
}

dataList.3 <- list(N=N, earnings=log(earnings), height=height,male=2-sex)
earnings_logmodel.sf1 <- sampling(earnings_multi_preds.sm, dataList.3)
print(earnings_logmodel.sf1)

## Including interactions (earnings_interactions.stan)
## lm(log.earn ~ height + male + height:male)
if (!file.exists("earnings_interactions.sm.RData")) {
    rt <- stanc("earnings_interactions.stan", model_name="earnings_interactions")
    earnings_interactions.sm <- stan_model(stanc_ret=rt)
    save(earnings_interactions.sm, file="earnings_interactions.sm.RData")
} else {
    load("earnings_interactions.sm.RData", verbose=TRUE)
}

earnings_interactions.sf1 <- sampling(earnings_interactions.sm, dataList.3)
print(earnings_interactions.sf1)

## Linear transformations (earnings_interaction_z.stan)
## lm(log.earn ~ z.height + male + z.height:male)
z.height <- (height - mean(height))/sd(height)
dataList.4 <- list(N=N, earnings=log(earnings), height=z.height,male=2-sex)
earnings_interactions.sf2 <- sampling(earnings_interactions.sm, dataList.4)
print(earnings_interactions.sf2)

## Log-log model (earnings_log_log.stan)
## lm(log.earn ~ log.height + male)
log.height <- log(height)
dataList.5 <- list(N=N, earnings=log(earnings), height=log.height,male=2-sex)
earnings_interactions.sf3 <- sampling(earnings_interactions.sm, dataList.5)
print(earnings_interactions.sf3)
