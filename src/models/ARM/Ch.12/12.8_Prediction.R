library(rstan)
library(ggplot2)
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

source("12.6_Group-LevelPredictors.R") # where variables were defined

## Prediction for a new observation in a new group (new house in county 26
## with x=1) FIXME: use stan
M2 <- lmer (y ~ x + u.full + (1 | county))
a.hat.M2 <- fixef(M2)[1] + fixef(M2)[3]*u + ranef(M2)$county
b.hat.M2 <- fixef(M2)[2]

x.tilde <- 1
sigma.y.hat <- sigma.hat(M2)$sigma$data
coef.hat <- as.matrix (coef(M2)$county)[26,]
y.tilde <- rnorm (1, coef.hat %*% c(1, x.tilde, u[26]), sigma.y.hat)
n.sims <- 1000
y.tilde <- rnorm (n.sims, coef.hat %*% c(1, x.tilde, u[26]), sigma.y.hat)

quantile (y.tilde, c(.25, .5, .75))

unlogged <- exp(y.tilde)
mean(unlogged)

## Prediction for a new observation in an existing group (new house in
## a new county)
u.tilde <- mean (u)
g.0.hat <- fixef(M2)["(Intercept)"]
g.1.hat <- fixef(M2)["u.full"]
sigma.a.hat <- sigma.hat(M2)$sigma$county

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

