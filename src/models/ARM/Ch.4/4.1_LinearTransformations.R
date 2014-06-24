library(rstan)
library(ggplot2)

### Data

source("earnings.data.R", echo = TRUE)

### First model: earn ~ height

data.list <- c("N", "earn", "height")
earn_height <- stan(file="earn_height.stan", data=data.list,
                    iter=1000, chains=4)
print(earn_height, pars = c("beta", "sigma", "lp__"))

### Figures

beta.post <- extract(earn_height, "beta")$beta
beta.mean <- colMeans(beta.post)
n <- 100
ndx <- sample(nrow(beta.post), n)
earnings.post <- data.frame(sampled.int = beta.post[ndx, 1],
                            sampled.slope = beta.post[ndx, 2],
                            id = ndx)
p <- ggplot(data.frame(height, earn), aes(x = height, y = earn)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.2)) +
    geom_abline(aes(intercept = sampled.int, slope = sampled.slope, group = id),
                data = earnings.post, alpha = 0.05) +
    geom_abline(aes(intercept = beta.mean[1], slope = beta.mean[2])) +
    theme(axis.text.y = element_text(angle = 90, hjust = 0.5)) +
    theme_bw()

# Figure 4.1 (left)
print(p +
      scale_x_continuous("height", breaks = seq(60, 75, 5)) +
      scale_y_continuous("earnings", breaks = c(0, 100000, 200000),
                         labels = c("0", "100000", "200000")) +
      ggtitle("Fitted linear model"))

## Figure 4.1 (right)
dev.new()
print(p +
      scale_x_continuous("height", limits = c(0, 80), breaks = seq(0, 80, 20)) +
      scale_y_continuous("earnings", limits = c(-200000, 200000),
                         breaks = c(-100000, 0, 100000),
                         labels = c("-100000", "0", "100000")) +
      ggtitle("x-axis extended to 0"))
