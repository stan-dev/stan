stopifnot(require(rstan))
library(ggplot2)
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### First model (kid_iq_one_pred.stan)
### lm(kid_score ~ mom_hs)

if (!file.exists("kid_iq_one_pred.sm.RData")) {
    rt <- stanc("kid_iq_one_pred.stan", model_name="kid_iq_one_pred")
    kid_iq_one_pred.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_one_pred.sm, file="kid_iq_one_pred.sm.RData")
} else {
    load("kid_iq_one_pred.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=length(kid_score), kid_score=kid_score, mom_pred=mom_hs)
kid_iq_one_pred.sf1 <- sampling(kid_iq_one_pred.sm, dataList.1)
print(kid_iq_one_pred.sf1)

### Second model: kid_score ~ mom_iq

dataList.2 <- list(N=length(kid_score), kid_score=kid_score, mom_pred=mom_iq)
kid_iq_one_pred.sf2 <- sampling(kid_iq_one_pred.sm, dataList.2)
print(kid_iq_one_pred.sf2)

# Figure 3.1
kidscore.jitter <- jitter(kid_score)
jitter.binary <- function(a, jitt=.05) {
   ifelse (a==0, runif (length(a), 0, jitt), runif (length(a), 1-jitt, 1))
}
jitter.mom_hs <- jitter.binary(mom_hs)

jitter.frame = data.frame(jitter.mom_hs=jitter.mom_hs,kidscore.jitter=kidscore.jitter)
m <- ggplot(jitter.frame,aes(x=jitter.mom_hs,y=kidscore.jitter))
m + geom_point() + scale_y_continuous("Mother Completed High School") + scale_x_continuous("Child Test Score") + theme_bw()

# Figure 3.2
# lm (kid_score ~ mom_iq)
beta.post <- extract(kid_iq_one_pred.sf2, "beta")$beta
beta.mean <- colMeans(beta.post)

kid.iq.one.pred.1 = data.frame(mom_iq=mom_iq,kid_score=kid_score)
m <- ggplot(kid.iq.one.pred.1,aes(x=mom_iq,y=kid_score))
m + geom_point() + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ score") + geom_abline(intercept = beta.mean[1], slope = beta.mean[2]) + theme_bw()
