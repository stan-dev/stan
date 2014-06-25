library(rstan)
library(ggplot2)

## Data

source("radon.data.R", echo = TRUE)

## Call Stan from R

radon.data <- c("N", "J", "y", "x", "county")
radon.1.sf <- stan(file='radon.1.stan', data=radon.data, iter = 500, chains=4)

# Plot Figure 16.2

mu_a.sample <- extract(radon.1.sf, pars = "mu_a",
                       permuted = FALSE, inc_warmup = TRUE)
n.chains <- dim(mu_a.sample)[2]
value <- matrix(mu_a.sample[1:200,,1], ncol = 1)
trace.ggdf <- data.frame(chain = rep(1:n.chains, each = 200),
                         iteration = rep(1:200, n.chains),
                         value)
p1 <- ggplot(trace.ggdf) +
    geom_path(aes(x = iteration, y = value, group = chain)) +
    ylab(expression(mu[alpha]))
print(p1)

## Accessing the simulations

sims <- extract(radon.1.sf)
a <- sims$a
b <- sims$b
sigma.y <- sims$sigma_y

# 90% CI for beta
quantile(b, c(0.05, 0.95))

# Prob. avg radon levels are higher in county 36 than in county 26
mean(a[,36] > a[,26])

## Fitted values, residuals and other calculations

a.multilevel <- rep(NA, J)
for (j in 1:J) {
    a.multilevel[j] <- median(a[,j])
}
b.multilevel <- median(b)

y.hat <- a.multilevel[county] + b.multilevel * x
y.resid <- y - y.hat

qplot(y.hat, y.resid)

# numeric calculations
n.sims <- 100
lqp.radon <- rep(NA, n.sims)
hennepin.radon <- rep(NA, n.sims)
for (s in 1:n.sims) {
  lqp.radon[s] <- exp(rnorm(1, a[s,36] + b[s], sigma.y[s]))
  hennepin.radon[s] <- exp(rnorm(1, a[s,26] + b[s], sigma.y[s]))
}
radon.diff <- lqp.radon - hennepin.radon
p2 <- ggplot(data.frame(radon.diff), aes(x = radon.diff)) +
    geom_histogram(color = "black", fill = "gray", binwidth = 0.75)
print(p2)
print(mean(radon.diff))
print(sd(radon.diff))
