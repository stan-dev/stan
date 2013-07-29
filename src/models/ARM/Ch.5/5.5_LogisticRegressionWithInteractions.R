library(rstan)
library(ggplot2)
source("wells.data.R", echo = TRUE)    

# Logistic regression with interactions (wells_interactions.stan)
# glm (switch ~ dist100 + arsenic + dist100:arsenic, family=binomial(link="logit"))
if (!exists("wells_interactions.sm")) {
    if (file.exists("wells_interactions.sm.RData")) {
        load("wells_interactions.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_interactions.stan", model_name = "wells_interactions")
        wells_interactions.sm <- stan_model(stanc_ret = rt)
        save(wells_interactions.sm, file = "wells_interactions.sm.RData")
    }
}
dist100 <- dist / 100
dataList.1 <- c("N","switc","dist","arsenic") 
wells_interactions.sf1 <- sampling(wells_interactions.sm, dataList.1)
print(wells_interactions.sf1)

beta.post <- extract(wells_interactions.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

## Centering the input variables 
c.dist <- dist - mean (dist)
c.arsenic <- arsenic - mean (arsenic)

## Refitting the model with centered inputs (wells_interactions_center.stan)
##  glm (switch ~ c.dist100 + c.arsenic + c.dist100:c.arsenic, family=binomial(link="logit"))
dataList.2 <- list(N=N, switc=switc, dist=c.dist,arsenic=c.arsenic)
wells_interactions_center.sf1 <- sampling(wells_interactions.sm, dataList.2)
print(wells_interactions_center.sf1)

## Graphing the model with interactions (Figure 5.12)
jitter.binary <- function(a, jitt=.05){
  ifelse (a==0, runif (length(a), 0, jitt), runif (length(a), 1-jitt, 1))
}

switch.jitter <- jitter.binary(switc)
frame3 = data.frame(dist=dist,switc=switch.jitter)
p1 <- ggplot(frame3,aes(x=dist,y=switc))  +
      geom_point() +
      scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) +
      scale_x_continuous("Distance (in meters) to nearest safe well") +
      theme_bw() +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean[1]+beta.mean[3]*0.5 + x * (beta.mean[2]/100 + 0.5 * beta.mean[4]/100))))) +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean[1]+beta.mean[3] + x * (beta.mean[2]/100 + beta.mean[4]/100)))))
print(p1)

dev.new()
frame4 = data.frame(ars=arsenic,switc=switch.jitter)
p2 <- ggplot(frame4,aes(x=ars,y=switc)) +
      geom_point() +
      scale_y_continuous("Pr(Switching)",limits=c(-.01,1)) +
      scale_x_continuous("Arsenic concentration in well water") +
      theme_bw() +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean[1]+beta.mean[3]*x)))) +
      stat_function(fun=function(x) 1.0 / (1 + exp(-(beta.mean[1]+beta.mean[2]*0.5 + x * (beta.mean[3] + beta.mean[4]*0.5)))))
print(p2)

# with community organization variable (wells_community.stan)
#  glm (switch ~ c.dist100 + c.arsenic + c.dist100:c.arsenic + assoc + educ4, family=binomial(link="logit"))
if (!exists("wells_community.sm")) {
    if (file.exists("wells_community.sm.RData")) {
        load("wells_community.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_community.stan", model_name = "wells_community")
        wells_community.sm <- stan_model(stanc_ret = rt)
        save(wells_community.sm, file = "wells_community.sm.RData")
    }
}
dataList.3 <- c("N","switc","dist","arsenic","assoc","educ")
wells_community.sf1 <- sampling(wells_community.sm, dataList.3)
print(wells_community.sf1)

# without community organization variable (wells_social.stan)
# glm (switch ~ c.dist100 + c.arsenic + c.dist100:c.arsenic + educ4, family=binomial(link="logit"))
if (!exists("wells_social.sm")) {
    if (file.exists("wells_social.sm.RData")) {
        load("wells_social.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_social.stan", model_name = "wells_social")
        wells_social.sm <- stan_model(stanc_ret = rt)
        save(wells_social.sm, file = "wells_social.sm.RData")
    }
}
dataList.4 <- c("N","switc","dist","arsenic","educ")
wells_social.sf1 <- sampling(wells_social.sm, dataList.4)
print(wells_social.sf1)

## Adding further interactions (centering education variable) (wells_interactions_center_educ.stan)
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
wells_interactions_center_educ.sf1 <- sampling(wells_interactions_center_educ.sm, dataList.3)
print(wells_interactions_center_educ.sf1)
