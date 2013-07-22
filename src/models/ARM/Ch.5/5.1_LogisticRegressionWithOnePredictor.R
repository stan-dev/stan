library(rstan)
library(ggplot2)
source("nes.data.R")    

 # Estimation (nes.stan)
 # glm (vote ~ income, family=binomial(link="logit"))

if (!file.exists("nes.sm.RData")) {
    rt <- stanc("nes.stan", model_name="nes")
    nes.sm <- stan_model(stanc_ret=rt)
    save(nes.sm, file="nes.sm.RData")
} else {
    load("nes.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, income=income, vote=vote)
nes.sf1 <- sampling(nes.sm, dataList.1)
print(nes.sf1)

fit1.post <- extract(nes.sf1)
beta.mean <- colMeans(fit1.post$beta)

 # Graph figure 5.1 (a)

frame1 = data.frame(income=income,vote=vote)
m <- ggplot(frame1,aes(x=income,y=vote))
m <- m + geom_point() + scale_y_continuous("Pr(Republican Vote)",limits=c(-.01,1)) + scale_x_continuous("Income",limits=c(-2,8)) + theme_bw() + stat_smooth(method="glm",family="binomial",se=F,size=2,colour="black") + geom_jitter(position=position_jitter(height=.08,width=.4)) + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(fit.1)[1] - coef(fit.1)[2] * x)))

 # Graph figure 5.1 (b) FIXME: loop doesn't work gives Warning Msg: Removed 562 rows containing missing values (geom_point). 

mm <- ggplot(frame1,aes(x=income,y=vote))
mm <- mm + scale_y_continuous("Pr(Republican Vote)",limits=c(-.01,1)) + scale_x_continuous("Income") + theme_bw() + stat_smooth(method="glm",family="binomial",se=F,colour="black") + geom_jitter(position=position_jitter(height=.08,width=.4)) + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(fit.1)[1] - coef(fit.1)[2] * x)))
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-1,1] - fit1.post$beta[4000-1,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-2,1] - fit1.post$beta[4000-2,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-3,1] - fit1.post$beta[4000-3,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-4,1] - fit1.post$beta[4000-4,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-5,1] - fit1.post$beta[4000-5,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-6,1] - fit1.post$beta[4000-6,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-7,1] - fit1.post$beta[4000-7,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-8,1] - fit1.post$beta[4000-8,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-9,1] - fit1.post$beta[4000-9,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-10,1] - fit1.post$beta[4000-10,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-11,1] - fit1.post$beta[4000-11,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-12,1] - fit1.post$beta[4000-12,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-13,1] - fit1.post$beta[4000-13,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-14,1] - fit1.post$beta[4000-14,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-15,1] - fit1.post$beta[4000-15,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-16,1] - fit1.post$beta[4000-16,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-17,1] - fit1.post$beta[4000-17,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-18,1] - fit1.post$beta[4000-18,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-19,1] - fit1.post$beta[4000-19,2] * x))},colour="grey")
mm +stat_function(fun=function(x) {1.0 / (1 + exp(-fit1.post$beta[4000-20,1] - fit1.post$beta[4000-20,2] * x))},colour="grey")
