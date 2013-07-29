library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

## Fitting the model (wells_interactions_center_educ.stan)
## glm (switch ~ c.dist100 + c.arsenic + c.educ4 + c.dist100:c.arsenic + c.dist100:c.educ4 + c.arsenic:c.educ4, family=binomial(link="logit"))
if (!exists("wells_interactions_center_educ.sm")) {
    if (file.exists("wells_interactions_center_educ.sm.RData")) {
        load("wells_interactions_center_educ.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_interactions_center_educ.stan", model_name = "wells_interactions_center_educ")
        wells_interactions_center_educ.sm <- stan_model(stanc_ret = rt)
        save(wells_interactions_center_educ.sm, file = "wells_interactions_center_educ.sm.RData")
    }
}
dataList.3 <- c("N","switc","dist","arsenic","assoc","educ")
wells_interactions_center_educ.sf1 <- sampling(wells_interactions_center_educ.sm, dataList.3)
print(wells_interactions_center_educ.sf1)

## Residual Plot (Figure 5.13 (a))
c.dist100 <- dist/100 - mean(dist)/100
c.arsenic <- arsenic - mean(arsenic)
c.educ4 <- educ/4 - mean(educ)/4
inter.dist.ars <- c.dist100 * c.arsenic
inter.dist.edu <- c.dist100 * c.educ4
inter.ars.edu <- c.arsenic * c.educ4

beta.post <- extract(wells_interactions_center_educ.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

pred <- 1.0 / (1.0 + exp(-(beta.mean[1] + beta.mean[2] * c.dist100 + beta.mean[3] * c.arsenic + beta.mean[4] * c.educ4 + beta.mean[5] * inter.dist.ars + beta.mean[6] * inter.dist.edu + beta.mean[7] * inter.ars.edu)))

frame1 = data.frame(sw=pred,res = switc-pred)
p1 <- ggplot(frame1,aes(x=sw,y=res)) + geom_point() +
      scale_y_continuous("Observed - Estimated") +
      scale_x_continuous("Estimated Pr(Switching)") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      labs(title="Residual Plot")
print(p1)

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
br.8 <- binned.resids (pred, switc-pred, nclass=40)$binned

dev.new()
frame2 = data.frame(x1=br.8[,1],y1=br.8[,2],x2=br.8[,6])
p2 <- ggplot(frame2,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Average residual") +
      scale_x_continuous("Distance to nearest safe well") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      geom_line(aes(x=x1,y=x2),colour="grey") +
      geom_line(aes(x=x1,y=-x2),colour="grey") +
      labs(title="Binned Residual Plot")
print(p2)

 ## Plot of binned residuals vs. inputs of interest

  # distance (Figure 5.14 (a))
dev.new()
br.dist <- binned.resids (dist, switc-pred, nclass=40)$binned
frame3 = data.frame(x1=br.dist[,1],y1=br.dist[,2],x2=br.dist[,6])
p3 <- ggplot(frame3,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Average residual") +
      scale_x_continuous("Distance to nearest safe well") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      geom_line(aes(x=x1,y=x2),colour="grey") +
      geom_line(aes(x=x1,y=-x2),colour="grey") +
      labs(title="Binned Residual Plot")
print(p3)

# arsenic (Figure 5.14 (b))
dev.new()
br.arsenic <- binned.resids (arsenic, switc-pred, nclass=40)$binned
frame4 = data.frame(x1=br.arsenic[,1],y1=br.arsenic[,2],x2=br.arsenic[,6])
p4 <- ggplot(frame4,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Average residual") +
      scale_x_continuous("Distance to nearest safe well") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      geom_line(aes(x=x1,y=x2),colour="grey") +
      geom_line(aes(x=x1,y=-x2),colour="grey") +
      labs(title="Binned Residual Plot")
print(p4)

## Log transformation (wells_log_transform.stan)
## glm (switch ~ c.dist100 + c.log.arsenic + c.educ4 + c.dist100:c.log.arsenic + c.dist100:c.educ4 + c.log.arsenic:c.educ4, family=binomial(link="logit"))
if (!exists("wells_log_transform.sm")) {
    if (file.exists("wells_log_transform.sm.RData")) {
        load("wells_log_transform.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_log_transform.stan", model_name = "wells_log_transform")
        wells_log_transform.sm <- stan_model(stanc_ret = rt)
        save(wells_log_transform.sm, file = "wells_log_transform.sm.RData")
    }
}
wells_log_transform.sf1 <- sampling(wells_log_transform.sm, dataList.3)
print(wells_log_transform.sf1)

## glm (switch ~ dist100 + log.arsenic + educ4 + dist100:log.arsenic + dist100:educ4 + log.arsenic:educ4, family=binomial(link="logit"))
if (!exists("wells_log_transform2.sm")) {
    if (file.exists("wells_log_transform2.sm.RData")) {
        load("wells_log_transform2.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_log_transform2.stan", model_name = "wells_log_transform2")
        wells_log_transform2.sm <- stan_model(stanc_ret = rt)
        save(wells_log_transform2.sm, file = "wells_log_transform2.sm.RData")
    }
}
wells_log_transform2.sf1 <- sampling(wells_log_transform2.sm, dataList.3)
print(wells_log_transform2.sf1)

## Graph for log model fit.9a (Figure 5.15 (a))
beta.post2 <- extract(wells_log_transform2.sf1, "beta")$beta
beta.mean2 <- colMeans(beta.post2)

jitter.binary <- function(a, jitt=.05){
  ifelse (a==0, runif (length(a), 0, jitt), runif (length(a), 1-jitt, 1))
}
dev.new()
switch.jitter <- jitter.binary(switc)
frame5 = data.frame(ars=arsenic,switc=switch.jitter)
p5 <- ggplot(frame5,aes(x=ars,y=switc)) +
      geom_point() +
      scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) +
      scale_x_continuous("Distance (in meters) to nearest safe well") +
      theme_bw() +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean2[1]+beta.mean2[3] * log(x) + beta.mean2[4] * mean(educ/4) + beta.mean2[7] * log(x) * mean(educ/4))))) +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean2[1]+beta.mean2[2]*0.5+beta.mean2[3]*log(x)+beta.mean2[4]*mean(educ/4)+beta.mean2[5]*0.5*log(x)+beta.mean2[6]*0.5*mean(educ/4)+beta.mean2[7]*log(x)*mean(educ/4)))))
print(p5)

## Graph of binned residuals for log model fit.9 (Figure 5.15 (b))
beta.post3 <- extract(wells_log_transform.sf1, "beta")$beta
beta.mean3 <- colMeans(beta.post3)

pred2 <- 1.0 / (1.0 + exp(-(beta.mean3[1] + beta.mean3[2] * (dist100 - mean(dist100)) + beta.mean3[3] * (log(arsenic) - mean(log(arsenic))) + beta.mean3[4] * (educ / 4.0 - mean(educ)/4.0) + beta.mean3[5] * (dist100 - mean(dist100)) * (log(arsenic) - mean(log(arsenic))) + beta.mean3[6] * (dist100 - mean(dist100)) * (educ/4.0 - mean(educ)/4.0) + beta.mean3[7] * (log(arsenic) - mean(log(arsenic))) * (educ/4.0 - mean(educ)/4.0))))

dev.new()
br.fit.9 <- binned.resids (arsenic, switc-pred2, nclass=40)$binned
frame4 = data.frame(x1=br.fit.9[,1],y1=br.fit.9[,2],x2=br.fit.9[,6])
p6 <- ggplot(frame4,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Average residual") +
      scale_x_continuous("Arsenic Level") +
      theme_bw() +
      geom_hline(yintercept=0,colour="grey") +
      geom_line(aes(x=x1,y=x2),colour="grey") +
      geom_line(aes(x=x1,y=-x2),colour="grey") +
      labs(title="Binned residual plot\nfor model with log (arsenic)")
print(p6)

## Error rates
 # in general
error.rate <- mean((predicted>0.5 & y==0) | (predicted<0.5 & y==1))

 # for model fit.9
error.rate <- mean((pred2>0.5 & switc==0) | (pred2<0.5 & switc==1))
