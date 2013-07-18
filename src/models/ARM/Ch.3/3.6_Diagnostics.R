stopifnot(require(rstan))
library(ggplot2)
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### Model (kid_iq_one_pred.stan)
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

 # Figure 3.12
beta.post <- extract(kid_iq_one_pred.sf2, "beta")$beta
beta.mean <- colMeans(beta.post)
resid <- kid_score - (beta.mean[1] + beta.mean[2] * mom_iq)
resid.sd <- sd(resid)

kid.iq.one.pred.1 = data.frame(mom_iq=mom_iq,resid=resid)
m <- ggplot(kid.iq.one.pred.1,aes(x=mom_iq,y=resid))
m + geom_point() + scale_y_continuous("Residuals") + scale_x_continuous("Mother IQ score") + geom_hline(yintercept=0,colour="grey70",size=1)  + geom_hline(yintercept = resid.sd, linetype="dashed",colour="grey70",size=1)  + geom_hline(yintercept = -resid.sd, linetype="dashed",colour="grey70",size=1) + theme_bw()
