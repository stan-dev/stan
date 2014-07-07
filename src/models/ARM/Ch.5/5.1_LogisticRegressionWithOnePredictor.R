library(rstan)
library(ggplot2)

### Data

source("nes1992_vote.data.R", echo = TRUE)

### Logistic model: vote ~ income
data.list <- c("N", "vote", "income")
nes_logit.sf <- stan(file='nes_logit.stan', data=data.list,
                     iter=1000, chains=4)
print(nes_logit.sf, pars = c("beta", "lp__"))

### Figures

beta.post <- extract(nes_logit.sf, "beta")$beta
beta.mean <- colMeans(beta.post)

# Figure 5.1 (a)

len <- 20
x <- seq(1, 5, length.out = len)
y <- 1 / (1 + exp(- beta.mean[1] - beta.mean[2] * x))
nes_vote.ggdf.1 <- data.frame(x, y)

p1 <- ggplot(data.frame(income, vote), aes(x = income, y = vote)) +
    geom_jitter(position = position_jitter(height = 0.04, width = 0.4)) +
    geom_line(aes(x, y), data = nes_vote.ggdf.1, size = 2) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- beta.mean[1] - beta.mean[2] * x))) +
    scale_x_continuous("Income", limits = c(-2, 8), breaks = seq(1, 5),
                       labels = c("1\n(poor)", "2", "3", "4", "5\n(rich)")) +
    scale_y_continuous("Pr(Republican Vote)", limits = c(-0.05, 1.05),
                       breaks = seq(0, 1, 0.2))
print(p1)

# Figure 5.1 (b)

dev.new()
n <- 20
ndx <- sample(nrow(beta.post), n)
min.x <- 0.5
max.x <- 5.5
x <- seq(min.x, max.x, length.out = len)
nes_vote.ggdf.2 <- data.frame(c(), c(), c()) # empty data frame
for (i in ndx) {
    y <- 1 / (1 + exp(- beta.post[i, 1] - beta.post[i, 2] * x))
    nes_vote.ggdf.2 <- rbind(nes_vote.ggdf.2,
                        data.frame(id = rep(i, len), x, y))
}

p2 <- ggplot(data.frame(income, vote), aes(x = income, y = vote)) +
    geom_jitter(position = position_jitter(height = .04, width = .4)) +
    geom_line(aes(x, y, group = id), data = nes_vote.ggdf.2, alpha = 0.1) +
    geom_line(aes(x, y = 1 / (1 + exp(- beta.mean[1] - beta.mean[2] * x))),
                  data = nes_vote.ggdf.2) +
    scale_x_continuous("Income", limits = c(min.x, max.x), breaks = seq(1, 5),
                       labels = c("1\n(poor)", "2", "3", "4", "5\n(rich)")) +
    scale_y_continuous("Pr(Republican Vote)", limits = c(-0.05, 1.05),
                       breaks = seq(0, 1, 0.2))
print(p2)
