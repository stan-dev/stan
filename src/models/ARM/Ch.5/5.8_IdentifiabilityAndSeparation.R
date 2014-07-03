library(rstan)
library(ggplot2)

### Data

# N <- 60
# x <- rnorm(N, mean = 1, sd = 2)
# y <- ifelse(x < 2, 0, 1)
# stan_rdump(c("N", "y", "x"), file = "separation.data.R")

source("separation.data.R", echo = TRUE)

## Model: y ~ x
data.list <- c("N", "y", "x")
separation.sf <- stan(file='separation.stan', data=data.list,
                      iter=1000, chains=4)
print(separation.sf)

## Plot
beta.post <- extract(separation.sf, "beta")$beta
b <- colMeans(beta.post)

p <- ggplot(data.frame(x, y), aes(x, y)) +
    geom_point() +
    stat_function(fun = function(x) 1.0 / (1 + exp(-(b[1] + b[2] * x))))
print(p)
