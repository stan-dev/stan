library(rstan)
library(ggplot2)
library(gridBase)
source("9.3_RandomizedExperiments.R") # where data was cleaned
## Treatment interactions and poststratification

 # model with only treat. indicator (electric_one_pred.stan)
 # lm (post.test ~ treatment, subset=(grade==4))
if (!exists("electric_one_pred.sm")) {
    if (file.exists("electric_one_pred.sm.RData")) {
        load("electric_one_pred.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_one_pred.stan", model_name = "electric_one_pred")
        electric_one_pred.sm <- stan_model(stanc_ret = rt)
        save(electric_one_pred.sm, file = "electric_one_pred.sm.RData")
    }
}

pt <- post.test(grade==4)
tr <- treatment(grade==4)
dataList.1 <- list(N=length(pt), post_test=pt, treatment=tr)
electric_one_pred.sf1 <- sampling(electric_one_pred.sm, dataList.1)
print(electric_one_pred.sf1)

 # model controlling for pre-test (electric_multi_preds.stan)
 # lm (post.test ~ treatment + pre.test, subset=(grade==4))
if (!exists("electric_multi_preds.sm")) {
    if (file.exists("electric_multi_preds.sm.RData")) {
        load("electric_multi_preds.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_multi_preds.stan", model_name = "electric_multi_preds")
        electric_multi_preds.sm <- stan_model(stanc_ret = rt)
        save(electric_multi_preds.sm, file = "electric_multi_preds.sm.RData")
    }
}
pt <- post.test[grade==4]
tr <- treatment[grade==4]
pr <- pre.test[grade==4]
dataList.2 <- list(N=length(pt), post_test=pt, treatment=tr,pre_test=pr)
electric_multi_preds.sf1 <- sampling(electric_multi_preds.sm, dataList.2)
print(electric_multi_preds.sf1)

## More than 2 treat. levels, continuous treat., multiple treat. factors (Figure 9.7)
if (!exists("electric_interactions.sm")) {
    if (file.exists("electric_interactions.sm.RData")) {
        load("electric_interactions.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_interactions.stan", model_name = "electric_interactions")
        electric_interactions.sm <- stan_model(stanc_ret = rt)
        save(electric_interactions.sm, file = "electric_interactions.sm.RData")
    }
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
  p1 <- ggplot() +
        geom_point(data=frame1,aes(x=x1,y=y1),shape=20) +
        geom_point(data=frame2,aes(x=x2,y=y2),shape=21) +
        scale_y_continuous("Posttest",limits=c(0,125)) +
        scale_x_continuous("Pretest",limits=c(0,125)) +
        theme_bw() +
        labs(title=paste("Grade ",j)) +
        geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3]+beta.mean[4]) +
        geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],linetype="dashed")
  print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = j))
}

## model to display uncertainty & Figure 9.8
electric_interactions.sf <- sampling(electric_interactions.sm, dataList.2)
print(electric_interactions.sf)
fit4.post <- extract(electric_interactions.sf)
beta.mean4 <- colMeans(fit4.post$beta)
dev.new()

p2 <- ggplot() +
      geom_point() +
      scale_y_continuous("Treatment Effect",limits=c(-5,10)) +
      scale_x_continuous("Pre-Test",limits=c(78.4,119.8)) +
      theme_bw() +
      labs(title="Treatment Effect in Grade 4")
for (i in 1:20) {
  p2 <- p2 + geom_abline(intercept=fit4.post$beta[4000-i,2],slope=fit4.post$beta[4000-i,4],colour="grey")
}
p2 + geom_abline(intercept=beta.mean4[2],slope=beta.mean4[4]) + geom_hline(yintercept=0,linetype="dashed")

 # compute the average treatment effect & summarize
n.sims <- nrow(fit4.post$beta)
effect <- array (NA, c(n.sims, sum(grade==4)))
for (i in 1:n.sims){
  effect[i,] <- fit4.post$beta[i,2] + fit4.post$beta[i,4]*pre.test[grade==4]
}
avg.effect <- rowMeans (effect)

print (c (mean(avg.effect), sd(avg.effect)))
