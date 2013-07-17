stopifnot(require(rstan))
library(ggplot2)
source("wells.data.R")    

## Fitting the model (wells_interactions_center_educ.stan)
if (!file.exists("wells_interactions_center_educ.sm.RData")) {
    rt <- stanc("wells_interactions_center_educ.stan", model_name="wells_interactions_center_educ")
    wells_interactions_center_educ.sm <- stan_model(stanc_ret=rt)
    save(wells_interactions_center_educ.sm, file="wells_all.sm.RData")
} else {
    load("wells_interactions_center_educ.sm.RData", verbose=TRUE)
}
dataList.3 <- list(N=N, switc=switc, dist=dist,arsenic=arsenic,assoc=assoc,educ=educ)
wells_interactions_center_educ.sf1 <- sampling(wells_interactions_center_educ.sm, dataList.3)
print(wells_interactions_center_educ.sf1)

## Residual Plot (Figure 5.13 (a))
c.dist100 <- dist/100 - mean(dist)/100
c.arsenic <- arsenic - mean(arsenic)
c.educ4 <- educ/4 - mean(educ)/4
inter.dist.ars <- c.dist100 * c.arsenic
inter.dist.edu <- c.dist100 * c.educ4
inter.ars.edu <- c.arsenic * c.educ4

switc.post <- extract(wells_interactions_center_educ.sf1, "switc")$
beta.mean <- colMeans(beta.post)

resid <- switc - (bet.mean[1] + beta.mean[2] * c.dist100 + beta.mean[3] * c.arsenic + beta.mean[4] * c.educ4 + beta.mean[5] * inter.dist.ars + beta.mean[6] * inter.dist.edu + beta.mean[7] * inter.ars.edu

frame1 = data.frame(switc=switc-resid,resid = resid)
m <- ggplot(frame1,aes(x=switc,y=resid))
m + geom_point() + scale_y_continuous("Observed - Estimated") + scale_x_continuous("Estimated Pr(Switching)") + theme_bw() + geom_hline(yintercept=0,colour="grey")

### Binned residual Plot 

 ## Defining binned residuals

binned.resids <- function (x, y, nclass=sqrt(length(x))){
  breaks.index <- floor(length(x)*(1:(nclass-1))/nclass)
  breaks <- c (-Inf, sort(x)[breaks.index], Inf)
  output <- NULL
  xbreaks <- NULL
  x.binned <- as.numeric (cut (x, breaks))
  for (i in 1:nclass){
    items <- (1:length(x))[x.binned==i]
    x.range <- range(x[items])
    xbar <- mean(x[items])
    ybar <- mean(y[items])
    n <- length(items)
    sdev <- sd(y[items])
    output <- rbind (output, c(xbar, ybar, n, x.range, 2*sdev/sqrt(n)))
  }
  colnames (output) <- c ("xbar", "ybar", "n", "x.lo", "x.hi", "2se")
  return (list (binned=output, xbreaks=xbreaks))
}

 ## Binned residuals vs. estimated probability of switching (Figure 5.13 (b))
br.8 <- binned.resids (switc-resid, resid, nclass=40)$binned

##FIXME weird lines



 ## Plot of binned residuals vs. inputs of interest

  # distance (Figure 5.14 (a))
br.dist <- binned.resids (dist, resid, nclass=40)$binned
##FIXME WEIRD LINES..
                  
  # arsenic (Figure 5.13 (b))
br.arsenic <- binned.resids (arsenic, resid, nclass=40)$binned
                  ##FIXME WEIRD LINES..

## Log transformation (wells_log_transform.stan)
if (!file.exists("wells_log_transform.sm.RData")) {
    rt <- stanc("wells_log_transform.stan", model_name="wells_log_transform")
    wells_log_transform.sm <- stan_model(stanc_ret=rt)
    save(wells_log_transform.sm, file="log_transform.sm.RData")
} else {
    load("wells_log_transform.sm.RData", verbose=TRUE)
}
wells_log_transform.sf1 <- sampling(wells_log_transform.sm, dataList.3)
print(wells_log_transform.sf1)

if (!file.exists("wells_log_transform2.sm.RData")) {
    rt <- stanc("wells_log_transform2.stan", model_name="wells_log_transform2")
    wells_log_transform2.sm <- stan_model(stanc_ret=rt)
    save(wells_log_transform2.sm, file="wells_log_transform2.sm.RData")
} else {
    load("wells_log_transform2.sm.RData", verbose=TRUE)
}
wells_log_transform2.sf1 <- sampling(wells_log_transform2.sm, dataList.3)
print(wells_log_transform2.sf1)

## Graph for log model fit.9a (Figure 5.15 (a))


## Graph of binned residuals for log model fit.9 (Figure 5.15 (b))


## Error rates

 # in general

error.rate <- mean((predicted>0.5 & y==0) | (predicted<0.5 & y==1))

 # for modell fit.9

error.rate <- mean((pred.9>0.5 & switch==0) | (pred.9<0.5 & switch==1))
