library(rstan)
library(ggplot2)

### Regression: weight ~ height

source("weight.data.R", echo = TRUE)

if (!exists("weight.sm")) {
    if (file.exists("weight.sm.RData")) {
        load("weight.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("weight.stan", model_name = "weight")
        weight.sm <- stan_model(stanc_ret = rt)
        save(weight.sm, file = "weight.sm.RData")
    }
}
weight.data <- c("N", "weight", "height")
censored.0.sf <- sampling(weight.sm, weight.data)
print(censored.0.sf)

### Censoring

source("weight_censored.data.R", echo = TRUE)

## Figure 18.8 (a)

weight_censored.ggdf <- data.frame(weight, height)

p1 <- ggplot(weight_censored.ggdf, aes(weight)) +
    geom_histogram(color = "black", fill = "gray", binwidth = 5) +
    xlab("weight measurement")
print(p1)

## Figure 18.8 (b)

dev.new()
p2 <- ggplot(weight_censored.ggdf, aes(height, weight)) +
    geom_jitter(position = position_jitter(width = 0.15, height = 1.5)) +
    ylab("weight measurement")
print(p2)

### Naive regression excluding the censored data

source("weight_lt200.data.R", echo = TRUE)
weight.data <- c("N", "weight", "height")
censored.1.sf <- sampling(weight.sm, weight.data)
print(censored.1.sf)

## Naive regression imputing the censoring points

source("weight_censored.data.R", echo = TRUE)
weight.data <- c("N", "weight", "height")
censored.2.sf <- sampling(weight.sm, weight.data)
print(censored.2.sf)

## Fitting the censored-data model

# source("weight_censored.data.R")
if (!exists("weight_censored.sm")) {
    if (file.exists("weight_censored.sm.RData")) {
        load("weight_censored.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("weight_censored.stan", model_name = "weight_censored")
        weight_censored.sm <- stan_model(stanc_ret = rt)
        save(weight_censored.sm, file = "weight_censored.sm.RData")
    }
}
censoring.data <- c("N", "N_obs", "N_cens", "C", "weight", "height")
censoring.sf <- sampling(weight_censored.sm, censoring.data)
print(censoring.sf, pars = c("a", "b", "sigma"))
