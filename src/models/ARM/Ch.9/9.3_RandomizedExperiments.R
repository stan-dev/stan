library(rstan)
library(ggplot2)
read.table ("electric.dat", header=T)

## Plot of the raw data (Figure 9.4)
for (j in 1:4){
  frame = data.frame(x1=control.Posttest[Grade==j])
  m1 <- ggplot(frame,aes(x=x1)) +  geom_histogram(colour = "black", fill = "white", binwidth=5) + theme_bw()
  print(m1)

  frame2 = data.frame(x1=treated.Posttest[Grade==j])
  m2 <- ggplot(frame2,aes(x=x1)) +  geom_histogram(colour = "black", fill = "white", binwidth=5) + theme_bw()
  print(m2)
}

## Basic analysis of a completely randomized experiment
source("electric.data.R")    
post.test <- c (treated.Posttest, control.Posttest)
pre.test <- c (treated.Pretest, control.Pretest)
grade <- rep (Grade, 2)
treatment <- rep (c(1,0), rep(length(treated.Posttest),2))
n <- length (post.test)


if (!file.exists("electric_one_pred.sm.RData")) {
    rt <- stanc("electric_one_pred.stan", model_name="electric_one_pred")
    electric_one_pred.sm <- stan_model(stanc_ret=rt)
    save(electric_one_pred.sm, file="electric_one_pred.sm.RData")
} else {
    load("electric_one_pred.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N1, post_test=post_test1, treatment=treatment1)
electric_one_pred.sf1 <- sampling(electric_one_pred.sm, dataList.1)
print(electric_one_pred.sf1)
beta.post1 <- extract(electric_one_pred.sf1, "beta")$beta
beta.mean1 <- colMeans(beta.post1)
sigma.post1 <- extract(electric_one_pred.sf1, "sigma")$sigma
sigma.mean1 <- mean(sigma.post1)

dataList.2 <- list(N=N2, post_test=post_test2, treatment=treatment2)
electric_one_pred.sf2 <- sampling(electric_one_pred.sm, dataList.2)
print(electric_one_pred.sf2)
beta.post2 <- extract(electric_one_pred.sf2, "beta")$beta
beta.mean2 <- colMeans(beta.post2)
sigma.post2 <- extract(electric_one_pred.sf2, "sigma")$sigma
sigma.mean2 <- mean(sigma.post2)

dataList.3 <- list(N=N3, post_test=post_test3, treatment=treatment3)
electric_one_pred.sf3 <- sampling(electric_one_pred.sm, dataList.3)
print(electric_one_pred.sf3)
beta.post3 <- extract(electric_one_pred.sf3, "beta")$beta
beta.mean3 <- colMeans(beta.post3)
sigma.post3 <- extract(electric_one_pred.sf3, "sigma")$sigma
sigma.mean3 <- mean(sigma.post3)

dataList.4 <- list(N=N4, post_test=post_test4, treatment=treatment4)
electric_one_pred.sf4 <- sampling(electric_one_pred.sm, dataList.4)
print(electric_one_pred.sf4)
beta.post4 <- extract(electric_one_pred.sf4, "beta")$beta
beta.mean4 <- colMeans(beta.post4)
sigma.post4 <- extract(electric_one_pred.sf4, "sigma")$sigma
sigma.mean4 <- mean(sigma.post4)

## Plot of the regression results (Figure 9.5) FIXME:CONDENSE TO ONE LOOP
if (!file.exists("electric_multi_preds.sm.RData")) {
    rt <- stanc("electric_multi_preds.stan", model_name="electric_multi_preds")
    electric_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(electric_multi_preds.sm, file="electric_multi_preds.sm.RData")
} else {
    load("electric_multi_preds.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N1, post_test=post_test1, treatment=treatment1,pre_test=pre_test1)
electric_multi_preds.sf1 <- sampling(electric_multi_preds.sm, dataList.1)
print(electric_multi_preds.sf1)
beta.post5 <- extract(electric_multi_preds.sf1, "beta")$beta
beta.mean5 <- colMeans(beta.post5)
sigma.post5 <- extract(electric_multi_preds.sf1, "sigma")$sigma
sigma.mean5 <- mean(sigma.post5)


dataList.2 <- list(N=N2, post_test=post_test2, treatment=treatment2,pre_test=pre_test2)
electric_multi_preds.sf2 <- sampling(electric_multi_preds.sm, dataList.2)
print(electric_multi_preds.sf2)
beta.post6 <- extract(electric_multi_preds.sf2, "beta")$beta
beta.mean6 <- colMeans(beta.post6)
sigma.post6 <- extract(electric_multi_preds.sf2, "sigma")$sigma
sigma.mean6 <- mean(sigma.post6)

dataList.3 <- list(N=N3, post_test=post_test3, treatment=treatment3,pre_test=pre_test3)
electric_multi_preds.sf3 <- sampling(electric_multi_preds.sm, dataList.3)
print(electric_multi_preds.sf3)
beta.post7 <- extract(electric_multi_preds.sf3, "beta")$beta
beta.mean7 <- colMeans(beta.post7)
sigma.post7 <- extract(electric_multi_preds.sf3, "sigma")$sigma
sigma.mean7 <- mean(sigma.post7)

dataList.4 <- list(N=N4, post_test=post_test4, treatment=treatment4,pre_test=pre_test4)
electric_multi_preds.sf4 <- sampling(electric_multi_preds.sm, dataList.4)
print(electric_multi_preds.sf4)
beta.post8 <- extract(electric_multi_preds.sf4, "beta")$beta
beta.mean8 <- colMeans(beta.post8)
sigma.post8 <- extract(electric_multi_preds.sf1, "sigma")$sigma
sigma.mean8 <- mean(sigma.post6)

 # function to make a graph out of the regression coeffs and se's
#FIXME..

#graphs on Figure 9.5 FIXME
 

## Controlling for pre-treatment predictors (Figure 9.6) FIXME:CHECK
for (j in 1:4){
  ok <- Grade==j
  pret <- c (treated.Pretest[ok], control.Pretest[ok])
  postt <-c (treated.Posttest[ok], control.Posttest[ok])
  t <- rep (c(1,0), rep(sum(ok),2))
  dataList.1 <- list(N=length(pret), post_test=postt, pre_test=pret,treatment=t)
  electric_multi_preds.sf <- sampling(electric_multi_preds.sm, dataList.1)
  beta.post <- extract(electric_multi_preds.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)

  frame1 = data.frame(x1=treated.Pretest[ok],y1=treated.Posttest[ok])
  frame2 = data.frame(x2=control.Pretest[ok],y2=control.Posttest[ok])
  m3 <- ggplot()
  m3 <- m3 + geom_point(data=frame1,aes(x=x1,y=y1),shape=20)
  m3 <- m3 + geom_point(data=frame2,aes(x=x2,y=y2),shape=21)
  m3 <- m3 + scale_y_continuous("Posttest",limits=c(0,125)) + scale_x_continuous("Pretest",limits=c(0,125)) + theme_bw() + labs(title=paste("Grade ",j))
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3])
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],linetype="dashed")
  print(m3)
}
