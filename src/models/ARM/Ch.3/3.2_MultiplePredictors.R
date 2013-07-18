library(rstan)
library(ggplot2)

### Data
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### Model (kid_iq_multi_preds.stan)
### lm (kid_score ~ mom_hs + mom_iq)

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

# Figure 3.3
ok <- (mom_hs==1)
frame1 = data.frame(mom_iq=mom_iq[ok],kid_score=kid_score[ok])
frame2 = data.frame(mom_iq=mom_iq[!ok],kid_score=kid_score[!ok])

beta.post <- extract(kid_iq_multi_preds.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

m <- ggplot() + geom_point(data=frame2,aes(x=mom_iq,y=kid_score)) + geom_point(data=frame1,aes(x=mom_iq,y=kid_score),colour="grey70")
m <- m + scale_y_continuous("Child Test Score") + scale_x_continuous("Mother IQ score")
m + geom_abline(intercept = beta.mean[1] + beta.mean[2], slope = beta.mean[3],colour = "grey70") + geom_abline(intercept = beta.mean[1], slope = beta.mean[3]) + theme_bw()
