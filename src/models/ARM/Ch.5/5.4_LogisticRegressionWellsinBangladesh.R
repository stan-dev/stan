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

data.list.1 <- c("N", "switched", "dist")
wells_dist.sf <- stan(file='wells_dist.stan', data=data.list.1,
                      iter=1000, chains=4)
print(wells_dist.sf, pars = c("beta", "lp__"))

# More reasonable model: switched ~ dist/100

wells_dist100.sf <- stan(file='wells_dist100.stan', data=data.list.1,
                         iter=1000, chains=4)
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

data.list.3 <- c("N", "switched", "dist", "arsenic")
wells_d100ars.sf <- stan(file='wells_d100ars.stan', data=data.list.3,
                         iter=1000, chains=4)
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
