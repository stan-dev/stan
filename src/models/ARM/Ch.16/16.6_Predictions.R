library(rstan)

## Data

source("radon.data.R", echo = TRUE)
radon.data <- c("N", "J", "y", "x", "county", "u")

## Predicting a new unit in an existing group using Stan

if (!exists("radon.2a.sm")) {
    if (file.exists("radon.2a.sm.RData")) {
        load("radon.2a.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.2a.stan", model_name = "radon.2a")
        radon.2a.sm <- stan_model(stanc_ret = rt)
        save(radon.2a.sm, file = "radon.2a.sm.RData")
    }
}
radon.2a.sf <- sampling(radon.2a.sm, radon.data)
print(radon.2a.sf)

y.tilde <- extract(radon.2a.sf, "y_tilde")$y_tilde
quantile(exp(y.tilde), c(.25, .75))

## Predicting a new unit in a new group using Stan

if (!exists("radon.2b.sm")) {
    if (file.exists("radon.2b.sm.RData")) {
        load("radon.2b.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.2b.stan", model_name = "radon.2b")
        radon.2b.sm <- stan_model(stanc_ret = rt)
        save(radon.2b.sm, file = "radon.2b.sm.RData")
    }
}
radon.2b.sf <- sampling(radon.2b.sm, radon.data)
print(radon.2b.sf)

y.tilde <- extract(radon.2b.sf, "y_tilde")$y_tilde
quantile(exp(y.tilde), c(.25, .75))

## Predictions using R

if (!exists("radon.2.sm")) {
    if (file.exists("radon.2.sm.RData")) {
        load("radon.2.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("radon.2.stan", model_name = "radon.2")
        radon.2.sm <- stan_model(stanc_ret = rt)
        save(radon.2.sm, file = "radon.2.sm.RData")
    }
}
radon.2.sf <- sampling(radon.2.sm, radon.data)
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

