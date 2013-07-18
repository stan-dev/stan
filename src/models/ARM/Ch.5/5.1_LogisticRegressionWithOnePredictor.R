stopifnot(require(rstan))
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

beta.post <- extract(nes.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

 # Graph figure 5.1 (a)

frame1 = data.frame(income=income,vote=vote)
m <- ggplot(frame1,aes(x=income,y=vote))
m <- m + geom_point() + scale_y_continuous("Pr(Republican Vote)",limits=c(-.01,1)) + scale_x_continuous("Income",limits=c(-2,8)) + theme_bw() + stat_smooth(method="glm",family="binomial",se=F,size=2,colour="black") + geom_jitter(position=position_jitter(height=.08,width=.4)) + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(fit.1)[1] - coef(fit.1)[2] * x)))

 # Graph figure 5.1 (b) FIXME: loop doesn't work gives Warning Msg: Removed 562 rows containing missing values (geom_point). 

fit.1 <- glm (vote ~ income, family=binomial(link="logit"))
sim.1 <- sim(fit.1)
mm <- ggplot(frame1,aes(x=income,y=vote))
mm <- mm + scale_y_continuous("Pr(Republican Vote)",limits=c(-.01,1)) + scale_x_continuous("Income") + theme_bw() + stat_smooth(method="glm",family="binomial",se=F,colour="black") + geom_jitter(position=position_jitter(height=.08,width=.4)) + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(fit.1)[1] - coef(fit.1)[2] * x)))
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[1,1] - coef(sim.1)[1,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[2,1] - coef(sim.1)[2,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[3,1] - coef(sim.1)[3,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[4,1] - coef(sim.1)[4,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[5,1] - coef(sim.1)[5,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[6,1] - coef(sim.1)[6,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[7,1] - coef(sim.1)[7,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[8,1] - coef(sim.1)[8,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[9,1] - coef(sim.1)[9,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[10,1] - coef(sim.1)[10,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[11,1] - coef(sim.1)[11,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[12,1] - coef(sim.1)[12,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[13,1] - coef(sim.1)[13,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[14,1] - coef(sim.1)[14,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[15,1] - coef(sim.1)[15,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[16,1] - coef(sim.1)[16,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[17,1] - coef(sim.1)[17,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[18,1] - coef(sim.1)[18,2] * x))},colour="grey")
  mm <- mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[19,1] - coef(sim.1)[19,2] * x))},colour="grey")
mm +stat_function(fun=function(x) {1.0 / (1 + exp(-coef(sim.1)[20,1] - coef(sim.1)[20,2] * x))},colour="grey")

