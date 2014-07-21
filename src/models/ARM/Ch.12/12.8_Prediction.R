library(rstan)
library(ggplot2)
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

source("12.6_Group-LevelPredictors.R") # where variables were defined

## Prediction for a new observation in a new group (new house in county 26
## with x=1)
dataList.3 <- list(N=length(y), y=y,x=x,county=county,u=u.full)
radon_group.sf1 <- stan(file='radon_group.stan', data=dataList.3,
                        iter=1000, chains=4)
print(radon_group.sf1)
post1 <- extract(radon_group.sf1)
post1.ranef <- colMeans(post1$const_coef)
mean.ranef <- mean(post1.ranef)
post1.beta <- colMeans(post1$beta)
post1.fixef1 <- mean(post1.ranef)

a.hat.M2 <- post1.fixef1 + post1.beta[2] * u + (post1.ranef - mean.ranef)
b.hat.M2 <- post1.beta[1]

x.tilde <- 1
sigma.y.hat <- mean(post1$sigma)
coef.hat <- as.matrix (post1$fixef1,post1$beta[1,26],post1$beta[2,26])
y.tilde <- rnorm (1, coef.hat %*% c(1, x.tilde, u[26]), sigma.y.hat)
n.sims <- 1000
y.tilde <- rnorm (n.sims, coef.hat %*% c(1, x.tilde, u[26]), sigma.y.hat)

quantile (y.tilde, c(.25, .5, .75))

unlogged <- exp(y.tilde)
mean(unlogged)

## Prediction for a new observation in an existing group (new house in
## a new county)
u.tilde <- mean (u)
g.0.hat <- post1.fixef1
g.1.hat <- post1.beta[2]
sigma.a.hat <- mean(post1$sigma)

a.tilde <- rnorm (n.sims, g.0.hat + g.1.hat*u.tilde, sigma.a.hat)
y.tilde <- rnorm (n.sims, a.tilde + b.hat*x.tilde, sigma.y.hat)

quantile (y.tilde, c(.25,.5,.75))

exp (quantile (y.tilde, c(.25,.5,.75)))

## Nonlinear predictions
y.tilde.basement <- rnorm (n.sims, a.hat.M2[26,], sigma.y.hat)
print (y.tilde.basement)

y.tilde.nobasement <- rnorm (n.sims, a.hat.M2[26,] + b.hat.M2, sigma.y.hat)
print (y.tilde.nobasement)

mean.radon.basement <- mean (exp (y.tilde.basement))
print (mean.radon.basement)

mean.radon.nobasement <- mean (exp (y.tilde.nobasement))
print (mean.radon.nobasement)

mean.radon <- .9*mean.radon.basement + .1*mean.radon.basement
print (mean.radon)

