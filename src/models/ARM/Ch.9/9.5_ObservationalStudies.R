library(rstan)
library(ggplot2)
read.table ("electric.dat", header=T)

## Observational studies FIXME CONVERT REGRESSION.2TABLSA

 # Plot Figure 9.9

 # function to make a graph out of the regression coeffs and se's

                  # graphs on Figure 9.9


## Plot of the regression results (Figure 9.5) FIXME:CONDENSE TO ONE LOOP
if (!file.exists("electric_multi_preds.sm.RData")) {
    rt <- stanc("electric_multi_preds.stan", model_name="electric_multi_preds")
    electric_multi_preds.sm <- stan_model(stanc_ret=rt)
    save(electric_multi_preds.sm, file="electric_multi_preds.sm.RData")
} else {
    load("electric_multi_preds.sm.RData", verbose=TRUE)
}

 

## Controlling for pre-treatment predictors (Figure 9.6)
for (j in 1:4){
  ok <- (grade==j) & (!is.na(supp))
  t <- rep (c(1,0), rep(sum(ok),2))
  frame1 = data.frame(x1=pre.test[ok],y1=post.test[ok])
  dataList.1 <- list(N=length(pret), post_test=post.test[ok], pre_test=pre.test[ok],treatment=supp)
  electric_multi_preds.sf <- sampling(electric_multi_preds.sm, dataList.1)
  beta.post <- extract(electric_multi_preds.sf, "beta")$beta
  beta.mean <- colMeans(beta.post)

  frame1 = data.frame(x1=treated.Pretest[ok],y1=treated.Posttest[ok])
  frame2 = data.frame(x2=control.Pretest[ok],y2=control.Posttest[ok])
  m3 <- ggplot()
  m3 <- m3 + geom_point(data=frame1,aes(x=x1,y=y1),shape=20)
  m3 <- m3 + scale_y_continuous("Posttest",limits=c(0,125)) + scale_x_continuous("Pretest",limits=c(0,125)) + theme_bw() + labs(title=paste("Grade ",j))
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]+beta.mean[2]),slope=beta.mean[3])
  m3 <- m3 + geom_abline(intercept=(beta.mean[1]),slope=beta.mean[3],colour="grey")
  print(m3)
}
