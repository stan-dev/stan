library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Predicting a new unit in an existing group using Stan

radon.2a.sf <- stan(file='radon.2a.stan', data=radon.data, iter=1000, chains=4)
print(radon.2a.sf)

y.tilde <- extract(radon.2a.sf, "y_tilde")$y_tilde
quantile(exp(y.tilde), c(.25, .75))

## Predicting a new unit in a new group using Stan

radon.2b.sf <- stan(file='radon.2b.stan', data=radon.data, iter=1000, chains=4)
print(radon.2b.sf)

y.tilde <- extract(radon.2b.sf, "y_tilde")$y_tilde
quantile(exp(y.tilde), c(.25, .75))

## Predictions using R

radon.2.sf <- stan(file='radon.2.stan', data=radon.data, iter=1000, chains=4)
sims <- extract(radon.2.sf)
# new unit in an existing group 
a <- sims$a
b <- sims$b
sigma.y <- sims$sigma_y
n.sims <- dim(a)[1]
y.tilde <- rnorm(n.sims, a[,26] + b * 1, sigma.y)
# new unit in a new group
g.0 <- sims$g_0
g.1 <- sims$g_1
u.tilde <- mean(u)
sigma.a <- sims$sigma_a
a.tilde <- rnorm(n.sims, g.0 + g.1 * u.tilde, sigma.a)
y.tilde <- rnorm(n.sims, a.tilde + b * 1, sigma.y)

