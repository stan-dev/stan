stopifnot(require(rstan))
library(ggplot2)

### Data
source("kid_iq.data.R")     # load kid_score, mom_hs, mom_iq

### Model: kid_score ~ mom_hs + mom_iq

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

## Figure 3.13
source("kid_iq2.data.R")     # load data

if (!file.exists("kid_iq_all.sm.RData")) {
    rt <- stanc("kid_iq_all.stan", model_name="kid_iq_all")
    kid_iq_all.sm <- stan_model(stanc_ret=rt)
    save(kid_iq_all.sm, file="kid_iq_all.sm.RData")
} else {
    load("kid_iq_all.sm.RData", verbose=TRUE)
}

dataList2 <- list(N=N, afqt=afqt, hs=hs, ppvt=ppvt)
kid_iq_all.sf <- sampling(kid_iq_all.sm, dataList2)
print(kid_iq_all.sf)

 # external validation
kid.iq.ev <- read.table("kid.iq.ext.val.txt", header=T) # Note: different data! Available at "Data for figure 3.13" file
kid.iq.ev$afqt <- (kid.iq.ev$afqt.adj-mean(kid.iq$afqt.adj))*(15/sqrt(var(kid.iq.ev$afqt.adj))) + 100
kid.iq.ev$hs <- as.numeric(kid.iq.ev$educ.cat!=1)

beta.post <- extract(kid_iq_all.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

cscores.new <- beta.mean[1] + beta.mean[2]*kid.iq.ev$hs + beta.mean[3]*kid.iq.ev$afqt
res <- (kid.iq.ev$ppvt - cscores.new)
a <- sqrt(var(res))

#left figure
frame1 = data.frame(cscores.new=cscores.new,ppvt=kid.iq.ev$ppvt)

m <- ggplot() + geom_point(data=frame1,aes(x=cscores.new,y=ppvt),colour="grey70")
m <- m + scale_y_continuous("Predicted Score",limits=c(20,140)) + scale_x_continuous("Actual Score",limits=c(20,140))
m + geom_abline(slope = 1) + theme_bw()

#right figure
frame2 = data.frame(cscores.new=cscores.new,res=res)

m <- ggplot() + theme_bw() + geom_point(data=frame2,aes(x=cscores.new,y=res),colour="grey70")
m <- m + scale_y_continuous("Predicted Score") + scale_x_continuous("Actual Score")
m + geom_hline(yintercept=0) + geom_hline(yintercept=a,linetype="dashed") + geom_hline(yintercept=-a,linetype="dashed")
