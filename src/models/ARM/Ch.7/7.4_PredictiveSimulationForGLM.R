library(rstan)
library(ggplot2)
library(arm)
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
beta.post <- extract(wells.sf1, "beta")$beta
beta.mean <- colMeans(beta.post)

 # Figure 7.6 (a)
fit.1 <- glm (switc ~ dist, family=binomial(link="logit"))
sim.1 <- sim (fit.1, n.sims=1000)

frame1 = data.frame(x1=coef(sim.1)[,1],y1=coef(sim.1)[,2])
m1 <- ggplot()
m1 <- m1 + geom_point()
m1 + theme_bw() + scale_y_continuous(expression(beta[1])) + scale_x_continuous(expression(beta[0])) 

 # Figure 7.6 (b)  FIXME LOOP IS BROKEN...
frame2 = data.frame(x1=dist,y1=switc)
m2 <- ggplot(frame2,aes(x=x1,y=y1))
m2 <- m2 + geom_point()
m2 <- m2 + stat_function(fun=function(x){ 1.0 / (1 + exp(-coef(sim.1)[1,1] - coef(sim.1)[1,2] * x))},colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[2,1] - coef(sim.1)[2,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[3,1] - coef(sim.1)[3,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[4,1] - coef(sim.1)[4,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[5,1] - coef(sim.1)[5,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[6,1] - coef(sim.1)[6,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[7,1] - coef(sim.1)[7,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[8,1] - coef(sim.1)[8,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[9,1] - coef(sim.1)[9,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[10,1] - coef(sim.1)[10,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[11,1] - coef(sim.1)[11,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[12,1] - coef(sim.1)[12,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[13,1] - coef(sim.1)[13,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[14,1] - coef(sim.1)[14,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[15,1] - coef(sim.1)[15,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[16,1] - coef(sim.1)[16,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[17,1] - coef(sim.1)[17,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[18,1] - coef(sim.1)[18,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[19,1] - coef(sim.1)[19,2] * x)),colour="grey")
m2 <- m2 + stat_function(fun=function(x) 1.0 / (1 + exp(-coef(sim.1)[20,1] - coef(sim.1)[20,2] * x)),colour="grey")
m2 + theme_bw() + scale_y_continuous("Pr(switching)") + scale_x_continuous("Distance (in meters) to the nearest safe well") + stat_function(fun=function(x) 1.0 / (1 + exp(-beta.mean[1] - beta.mean[2] * x)))

## Predictive simulation using the binomial distribution

n.sims <- 1000
X.tilde <- cbind (1, dist)
n.tilde <- nrow (X.tilde)
y.tilde <- array (NA, c(n.sims, n.tilde))
for (s in 1:n.sims){
  p.tilde <- invlogit (X.tilde %*% sim.1$coef[s,])
  y.tilde <- rbinom (n.tilde, 1, p.tilde)
}

## Predictive simulation using latent logistic distribution

logit <- function (a) {log(a/(1-a))}

y.tilde <- array (NA, c(n.sims, n.tilde))
for (s in 1:n.sims){
  epsilon.tilde <- logit (runif (n.tilde, 0, 1))
  z.tilde <- X.tilde %*% sim.1$coef[s,] + epsilon.tilde
  y.tilde[s,] <- ifelse (z.tilde>0, 1, 0)
}

# Alternative using matrix algebra

epsilon.tilde <- array (logit (runif (n.sims*n.tilde, 0, 1)),
                        c(n.sims, n.tilde))
z.tilde <- sim.1$coef %*% t(X.tilde) + epsilon.tilde
y.tilde <- ifelse (z.tilde>0, 1, 0)


### Compound models

 ## Models
source("earnings1.data.R") ##FIXME. SWITCH TO STAN WHEN SIM SORTED OUT
fit.1a <- glm (earn_pos ~ height + male, family=binomial(link="logit"))
source("earnings.data.R") ##FIXME. SWITCH TO STAN WHEN SIM SORTED OUT
male <- 2 - sex1
log.earn <- log (earnings)
fit.1b <- lm (log.earn ~ height + male, subset=earnings>0)

x.new <- c (1, 68, 1)          # constant term=1, height=68, male=1

 # Simulation ignoring uncertainty

n.sims <- 1000
prob.earn.pos <- invlogit (coef(fit.1a) %*% x.new)
earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)
earn.sim <- ifelse (earn.pos.sim==0, 0, 
  exp (rnorm (n.sims, coef(fit.1b) %*% x.new, sigma.hat(fit.1b))))

 # Simulated values of coefficient estimates

sim.1a <- sim (fit.1a, n.sims)
sim.1b <- sim (fit.1b, n.sims)
prob.earn.pos <- invlogit (coef(sim.1a) %*% x.new)
earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)
earn.sim <- ifelse (earn.pos.sim==0, 0,
  exp (rnorm (n.sims, coef(sim.1b) %*% x.new, sigma(sim.1b))))

  # Computations into a function

Mean.earn <- function (height, male, sim.a, sim.b){
  x.new <- c (1, height, male)
  prob.earn.pos <- invlogit (coef(sim.a)%*% x.new)
  earn.pos.sim <- rbinom (n.sims, 1, prob.earn.pos)  
  earn.sim <- ifelse (earn.pos.sim==0, 0,
    exp (rnorm (n.sims, coef(sim.b) %*% x.new, sigma(sim.b))))
  return (mean (earn.sim))
}

heights <- seq (60, 75, 1)
mean.earn.female <- sapply (heights, Mean.earn, male=0, sim.1a, sim.1b)
mean.earn.male <- sapply (heights, Mean.earn, male=1, sim.1a, sim.1b)

  # or

heights <- seq (60, 75, 1)
k <- length (heights) 
mean.earn.female <- rep (NA, k)
mean.earn.male <- rep (NA, k)
for (i in 1:k){
  mean.earn.female[i] <- Mean.earn (heights[i], 0, sim.1a, sim.1b)
  mean.earn.male[i] <- Mean.earn (heights[i], 1, sim.1a, sim.1b)
}

frame = data.frame(x1=heights,x2=mean.earn.female)
m <- ggplot(frame,aes(y=x2,x=x1))
m + theme_bw() + geom_point()
