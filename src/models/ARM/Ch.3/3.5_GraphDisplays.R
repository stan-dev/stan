library(rstan)
library(ggplot2)
library(arm)
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### First model (kid_iq_one_pred.stan)
### lm (kid_score ~ mom_iq)

if (!file.exists("kid_iq_one_pred.sm.RData")) {
    rt <- stanc("kid_iq_one_pred.stan", model_name="kid_iq_one_pred")
    kid_iq_one_pred.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_one_pred.sm, file="kid_iq_one_pred.sm.RData")
} else {
    load("kid_iq_one_pred.sm.RData", verbose=TRUE)
}

dataList.2 <- list(N=length(kid_score), kid_score=kid_score, mom_pred=mom_iq)
kid_iq_one_pred.sf2 <- sampling(kid_iq_one_pred.sm, dataList.2)
print(kid_iq_one_pred.sf2)

## Regression line as a function of one input variable
beta.post <- extract(kid_iq_one_pred.sf2, "beta")$beta
beta.mean1 <- colMeans(beta.post)

kid.iq.one.pred.1 = data.frame(mom_iq=mom_iq,kid_score=kid_score)
m <- ggplot(kid.iq.one.pred.1,aes(x=mom_iq,y=kid_score))
m + geom_point() + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ score") + geom_abline(intercept = beta.mean1[1], slope = beta.mean1[2]) + theme_bw()

## model with no interaction (kid_iq_multi_preds.stan)
## lm (kid_score ~ mom_hs + mom_iq)
if (!file.exists("kid_iq_multi_preds.sm.RData")) {
    rt <- stanc("kid_iq_multi_preds.stan", model_name="kid_iq_multi_preds")
    kid_iq_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_multi_preds.sm, file="kid_iq_multi_preds.sm.RData")
} else {
    load("kid_iq_multi_preds.sm.RData", verbose=TRUE)
}

dataList <- list(N=N, kid_score=kid_score, mom_hs=mom_hs, mom_iq=mom_iq)
kid_iq_multi_preds.sf <- sampling(kid_iq_multi_preds.sm, dataList)
print(kid_iq_multi_preds.sf)

#Graphics for above model
ok <- (mom_hs==1)
frame1 = data.frame(mom_iq=mom_iq[ok],kid_score=kid_score[ok])
frame2 = data.frame(mom_iq=mom_iq[!ok],kid_score=kid_score[!ok])

beta.post <- extract(kid_iq_multi_preds.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

m <- ggplot() + geom_point(data=frame2,aes(x=mom_iq,y=kid_score)) + geom_point(data=frame1,aes(x=mom_iq,y=kid_score),colour="grey70")
m <- m + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ score")
m + geom_abline(intercept = beta.mean[1] + beta.mean[2], slope = beta.mean[3],colour = "grey70") + geom_abline(intercept = beta.mean[1], slope = beta.mean[3]) + theme_bw()

## model with interaction (kid_iq_interaction.stan)
## lm (kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq)

if (!file.exists("kid_iq_interaction.sm.RData")) {
    rt <- stanc("kid_iq_interaction.stan", model_name="kid_iq_interaction")
    kid_iq_interaction.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_interaction.sm, file="kid_iq_interaction.sm.RData")
} else {
    load("kid_iq_interaction.sm.RData", verbose=TRUE)
}

dataList <- list(N=N, kid_score=kid_score, mom_hs=mom_hs, mom_iq=mom_iq)
kid_iq_interaction.sf <- sampling(kid_iq_interaction.sm, dataList)
print(kid_iq_interaction.sf)

#Graphics for above model
ok <- (mom_hs==1)
frame1 = data.frame(mom_iq=mom_iq[ok],kid_score=kid_score[ok])
frame2 = data.frame(mom_iq=mom_iq[!ok],kid_score=kid_score[!ok])

beta.post <- extract(kid_iq_interaction.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

m <- ggplot() + geom_point(data=frame2,aes(x=mom_iq,y=kid_score)) + geom_point(data=frame1,aes(x=mom_iq,y=kid_score),colour="grey70")
m <- m + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ score")
m + geom_abline(intercept = beta.mean[1] + beta.mean[2], slope = beta.mean[3] + beta.mean[4],colour = "grey70") + geom_abline(intercept = beta.mean[1], slope = beta.mean[3]) + theme_bw()

### Displaying uncertainty in the fitted regression (Figure 3.10)
fit.2 <- lm (kid_score ~ mom_iq)
display(fit.2)

fit.2.sim <- sim (fit.2)
frame1 = data.frame(ks=kid_score,miq=mom_iq)
m <- ggplot(frame1,aes(x=miq,y=ks))
m <- m + geom_point() + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ Score") + theme_bw()
for (i in 1:10)
  m <- m + geom_abline(intercept=coef(fit.2.sim)[i,1],slope=coef(fit.2.sim)[i,2],colour="grey",size=2)
m + geom_abline(intercept=beta.mean1[1],slope=beta.mean1[2],colour="red")

### Displaying using one plot for each input variable (Figure 3.11) FIXME

