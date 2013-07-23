library(rstan)
library(ggplot2)
source("wells.data.R")    

## Logistic regression (wells.stan)
## glm (switch ~ dist, family=binomial(link="logit"))

if (!file.exists("wells.sm.RData")) {
    rt <- stanc("wells.stan", model_name="wells")
    wells.sm <- stan_model(stanc_ret=rt)
    save(wells.sm, file="wells.sm.RData")
} else {
    load("wells.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, dist=dist, switc=switc)
wells.sf1 <- sampling(wells.sm, dataList.1)
print(wells.sf1)
fit1.post <- extract(wells.sf1)
beta.mean <- colMeans(fit1.post$beta)

 # Figure 7.6 (a)

frame1 = data.frame(x1=coef(sim.1)[,1],y1=coef(sim.1)[,2])
m1 <- ggplot()
m1 <- m1 + geom_point()
m1 + theme_bw() + scale_y_continuous(expression(beta[1])) + scale_x_continuous(expression(beta[0])) 

 # Figure 7.6 (b)
frame2 = data.frame(x1=dist,y1=switc)
m2 <- "ggplot(frame2,aes(x=x1,y=y1)) + geom_point()+ theme_bw() + scale_y_continuous('Pr(switching)') + scale_x_continuous('Distance (in meters) to the nearest safe well')"
for (i in 1:20) {
  m2 <- paste(m2,"+ stat_function(aes(y=0),fun=function(x) 1.0 / (1 + exp(-fit1.post$beta[4000-",i,",1]-fit1.post$beta[4000-",i,",2]*x)),colour='grey')")
}
m2 <- paste(m2, "+ stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean[1] - beta.mean[2] * x)))")
eval(parse(text = m2))

## Predictive simulation using the binomial distribution

n.sims <- 4000
X.tilde <- cbind (1, dist)
n.tilde <- nrow (X.tilde)
y.tilde <- array (NA, c(n.sims, n.tilde))
for (s in 1:n.sims){
  p.tilde <- invlogit (X.tilde %*% fit1.post$beta[s,])
  y.tilde <- rbinom (n.tilde, 1, p.tilde)
}

## Predictive simulation using latent logistic distribution

logit <- function (a) {log(a/(1-a))}

y.tilde <- array (NA, c(n.sims, n.tilde))
for (s in 1:n.sims){
  epsilon.tilde <- logit (runif (n.tilde, 0, 1))
  z.tilde <- X.tilde %*% fit1.post$beta[s,] + epsilon.tilde
  y.tilde[s,] <- ifelse (z.tilde>0, 1, 0)
}

# Alternative using matrix algebra

epsilon.tilde <- array (logit (runif (n.sims*n.tilde, 0, 1)),
                        c(n.sims, n.tilde))
z.tilde <- fit1.post$beta %*% t(X.tilde) + epsilon.tilde
y.tilde <- ifelse (z.tilde>0, 1, 0)


### Compound models

## Models (earnings1.stan)
## glm (earn_pos ~ height + male, family=binomial(link="logit"))
source("earnings1.data.R")

if (!file.exists("earnings1.sm.RData")) {
    rt <- stanc("earnings1.stan", model_name="earnings1")
    earnings1.sm <- stan_model(stanc_ret=rt)
    save(earnings1.sm, file="earnings1.sm.RData")
} else {
    load("earnings1.sm.RData", verbose=TRUE)
}

dataList.2 <- list(N=N, earn_pos=earn_pos, height=height,male=male)
earnings1.sf1 <- sampling(earnings1.sm, dataList.2)
print(earnings1.sf1)
fit1a.post <- extract(earnings1.sf1)

## (earnings2.stan)
##model lm (log.earn ~ height + male, subset=earnings>0)
source("earnings2.data.R")

if (!file.exists("earnings2.sm.RData")) {
    rt <- stanc("earnings2.stan", model_name="earnings2")
    earnings2.sm <- stan_model(stanc_ret=rt)
    save(earnings1.sm, file="earnings2.sm.RData")
} else {
    load("earnings2.sm.RData", verbose=TRUE)
}

dataList.3 <- list(N=N, earnings=earnings, height=height,sex=sex)
earnings2.sf1 <- sampling(earnings2.sm, dataList.3)
print(earnings2.sf1)
fit1b.post <- extract(earnings2.sf1)

x.new <- c (1, 68, 1)          # constant term=1, height=68, male=1

 # Simulation ignoring uncertainty

n.sims <- 4000
prob.earn.pos <- invlogit (fit1a.post$beta %*% x.new)
earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)
earn.sim <- ifelse (earn.pos.sim==0, 0, 
  exp (rnorm (n.sims, fit1a.post$beta %*% x.new,mean(fit1b.post$sigma))))

 # Simulated values of coefficient estimates

sim.1a <- sim (fit.1a, n.sims)
sim.1b <- sim (fit.1b, n.sims)
prob.earn.pos <- invlogit (fit1a.post$beta %*% x.new)
earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)
earn.sim <- ifelse (earn.pos.sim==0, 0,
  exp (rnorm (n.sims, fit1b.post$beta %*% x.new, fit1b.post$sigma)))

# Computations into a function

Mean.earn <- function (height, male, fit1a.post, fit1b.post){
  x.new <- c (1, height, male)
  prob.earn.pos <- invlogit (fit1a.post$beta%*% x.new)
  earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)  
  earn.sim <- ifelse (earn.pos.sim==0, 0,
    exp (rnorm (n.sims, fit1b.post$beta %*% x.new, fit1b.post$sigma)))
  return (mean (earn.sim))
}

heights <- seq (60, 75, 1)
mean.earn.female <- sapply (heights, Mean.earn, male=0, fit1a.post, fit1b.post)
mean.earn.male <- sapply (heights, Mean.earn, male=1, fit1a.post, fit1b.post)

frame = data.frame(x1=heights,x2=mean.earn.female)
m <- ggplot(frame,aes(y=x2,x=x1))
m + theme_bw() + geom_point()
