library(rstan)
library(ggplot2)

### Data

source("wells.data.R", echo = TRUE)

### Logistic regression with one predictor

# Figure 5.8

p1 <- ggplot(data.frame(dist)) +
    geom_histogram(aes(dist), color = "black", fill = "gray", binwidth = 10) +
    scale_x_continuous("Distance (in meters) to the nearest safe well") +
    scale_y_continuous("")
print(p1)

# First logistic model: switched ~ dist

if (!exists("wells_dist.sm")) {
    if (file.exists("wells_dist.sm.RData")) {
        load("wells_dist.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dist.stan", model_name = "wells_dist")
        wells_dist.sm <- stan_model(stanc_ret = rt)
        save(wells_dist.sm, file = "wells_dist.sm.RData")
    }
}

data.list.1 <- c("N", "switched", "dist")
wells_dist.sf <- sampling(wells_dist.sm, data.list.1)
print(wells_dist.sf, pars = c("beta", "lp__"))

# More reasonable model: switched ~ dist/100

if (!exists("wells_dist100.sm")) {
    if (file.exists("wells_dist100.sm.RData")) {
        load("wells_dist100.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dist100.stan", model_name = "wells_dist100")
        wells_dist100.sm <- stan_model(stanc_ret = rt)
        save(wells_dist100.sm, file = "wells_dist100.sm.RData")
    }
}

wells_dist100.sf <- sampling(wells_dist100.sm, data.list.1)
print(wells_dist100.sf, pars = c("beta", "lp__"))

# Figure 5.9

beta.post.2 <- extract(wells_dist100.sf, "beta")$beta
beta.mean.2 <- colMeans(beta.post.2)

dev.new()
p2 <- ggplot(data.frame(switched, dist), aes(dist, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean.2[1] - beta.mean.2[2] * x / 100))) +
    scale_x_continuous("Distance (in meters) to the nearest safe well",
                       breaks = seq(from = 0, by = 50, length.out = 7)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
print(p2)

### Logistic regression with second input variable

# Figure 5.10

dev.new()
p3 <- ggplot(data.frame(arsenic)) +
    geom_histogram(aes(arsenic), color = "black", fill = "gray", binwidth = 0.25) +
    scale_x_continuous("Arsenic concentration in well water") +
    scale_y_continuous("")
print(p3)

# Model: switched ~ dist/100 + arsenic

if (!exists("wells_d100ars.sm")) {
    if (file.exists("wells_d100ars.sm.RData")) {
        load("wells_d100ars.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_d100ars.stan", model_name = "wells_d100ars")
        wells_d100ars.sm <- stan_model(stanc_ret = rt)
        save(wells_d100ars.sm, file = "wells_d100ars.sm.RData")
    }
}

data.list.3 <- c("N", "switched", "dist", "arsenic")
wells_d100ars.sf <- sampling(wells_d100ars.sm, data.list.3)
print(wells_d100ars.sf, pars = c("beta", "lp__"))

beta.post.3 <- extract(wells_d100ars.sf, "beta")$ beta
beta.mean.3 <- colMeans(beta.post.3)

# Figure 5.11 (a)

dev.new()
p4 <- ggplot(data.frame(switched, dist), aes(dist, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean.3[1] - beta.mean.3[2] * x / 100
                       - beta.mean.3[3] * 0.5))) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean.3[1] - beta.mean.3[2] * x / 100
                       - beta.mean.3[3]))) +
    annotate("text", x = c(50,75), y = c(0.35, 0.55),
             label = c("if As = 0.5", "if As = 1.0"), size = 4) +
    scale_x_continuous("Distance (in meters) to the nearest safe well",
                       breaks = seq(from = 0, by = 50, length.out = 7)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
plot(p4)

# Figure 5.11 (b)

dev.new()
p5 <- ggplot(data.frame(switched, arsenic), aes(arsenic, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean.3[1] - beta.mean.3[3] * x))) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean.3[1] - beta.mean.3[2] * 0.5
                       - beta.mean.3[3] * x))) +
    annotate("text", x = c(1.7,2.5), y = c(0.78, 0.56),
             label = c("if dist = 0", "if dist = 50"), size = 4) +
    scale_x_continuous("Arsenic concentration in well water",
                       breaks = seq(from = 0, by = 2, length.out = 5)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
print(p5)
