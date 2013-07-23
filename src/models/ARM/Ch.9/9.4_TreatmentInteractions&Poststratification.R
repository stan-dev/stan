library(rstan)
library(ggplot2)
library(gridBase)
source("9.3_RandomizedExperiments.R") # where data was cleaned
##FIXME CHECK THAT THSE WORK
## Treatment interactions and poststratification

 # model with only treat. indicator (electric_one_pred.stan)
 # lm (post.test ~ treatment, subset=(grade==4))
if (!file.exists("electric_one_pred.sm.RData")) {
    rt <- stanc("electric_one_pred.stan", model_name="electric_one_pred")
    electric_one_pred.sm <- stan_model(stanc_ret=rt)
    save(electric_one_pred.sm, file="electric_one_pred.sm.RData")
} else {
    load("electric_one_pred.sm.RData", verbose=TRUE)
}

pt <- post.test(grade==4)
tr <- treatment(grade==4)
dataList.1 <- list(N=length(pt), post_test=pt, treatment=tr)
electric_one_pred.sf1 <- sampling(electric_one_pred.sm, dataList.1)
print(electric_one_pred.sf1)

 # model controlling for pre-test (electric_multi_preds.stan)
 # lm (post.test ~ treatment + pre.test, subset=(grade==4))
if (!file.exists("electric_multi_preds.sm.RData")) {
    rt <- stanc("electric_multi_preds.stan", model_name="electric_multi_preds")
    electric_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(electric_multi_preds.sm, file="electric_multi_preds.sm.RData")
} else {
    load("electric_multi_preds.sm.RData", verbose=TRUE)
}
pt <- post.test[grade==4]
tr <- treatment[grade==4]
pr <- pre.test[grade==4]
dataList.2 <- list(N=length(pt), post_test=pt, treatment=tr,pre_test=pr)
electric_multi_preds.sf1 <- sampling(electric_multi_preds.sm, dataList.2)
print(electric_multi_preds.sf1)

## More than 2 treat. levels, continuous treat., multiple treat. factors (Figure 9.7)
if (!file.exists("electric_interactions.sm.RData")) {
    rt <- stanc("electric_interactions.stan", model_name="electric_interactions")
    electric_interactions.sm <- stan_model(stanc_ret=rt)
    save(electric_interactions.sm, file="electric_interactions.sm.RData")
} else {
    load("electric_interactions.sm.RData", verbose=TRUE)
}

electric <- read.table ("electric.dat", header=T)
attach(electric)
pushViewport(viewport(layout = grid.layout(1, 4)))
for (j in 1:4){
  ok <- electric$Grade==j & !is.na(treated.Posttest+treated.Pretest+control.Pretest+control.Posttest)
  pret <- c (treated.Pretest[ok], control.Pretest[ok])
  postt <-c (treated.Posttest[ok], control.Posttest[ok])
  t <- rep (c(1,0),rep(sum(ok),2))
  dataList.1 <- list(N=length(pret), post_test=postt, pre_test=pret,treatment=t)
  electric_interactions.sf <- sampling(electric_interactions.sm, dataList.1)
  beta.post <- extract(electric_interactions.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)

  frame1 = data.frame(x1=treated.Pretest[ok],y1=treated.Posttest[ok])
  frame2 = data.frame(x2=control.Pretest[ok],y2=control.Posttest[ok])
  m3 <- ggplot()
  m3 <- m3 + geom_point(data=frame1,aes(x=x1,y=y1),shape=20)
  m3 <- m3 + geom_point(data=frame2,aes(x=x2,y=y2),shape=21)
  m3 <- m3 + scale_y_continuous("Posttest",limits=c(0,125)) + scale_x_continuous("Pretest",limits=c(0,125)) + theme_bw() + labs(title=paste("Grade ",j))
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3]+beta.mean[4])
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],linetype="dashed")
  print(m3, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

## model to display uncertainty & Figure 9.8
electric_interactions.sf <- sampling(electric_interactions.sm, dataList.2)
print(electric_interactions.sf)
beta.post <- extract(electric_interactions.sf, "beta")$beta
beta.mean4 <- colMeans(beta.post)

 lm.4 <- lm (post.test ~ tr + pr + tr:pr) #grade==4
 lm.4.sim <- sim (lm.4)
m <- ggplot()
m <- m + geom_point() + scale_y_continuous("Treatment Effect") + scale_x_continuous("Pre-Test") + theme_bw() + labs(title="Treatment Effect in Grade 4")
for (i in 1:10)
  m <- m + geom_abline(intercept=coef(lm.4.sim)[i,2],slope=coef(lm.4.sim)[i,4],colour="grey",size=2)
m + geom_abline(intercept=beta.mean4[2],slope=beta.mean4[4]) + geom_hline(yintercept=0,linetype="dashed")

 # compute the average treatment effect & summarize
n.sims <- nrow(lm.4.sim$coef)
effect <- array (NA, c(n.sims, sum(grade==4)))
for (i in 1:n.sims){
  effect[i,] <- lm.4.sim@coef[i,2] + lm.4.sim@coef[i,4]*pre.test[grade==4]
}
avg.effect <- rowMeans (effect)

print (c (mean(avg.effect), sd(avg.effect)))
