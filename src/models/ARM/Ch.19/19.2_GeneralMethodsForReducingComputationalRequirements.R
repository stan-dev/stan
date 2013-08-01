library(rstan)
library(ggplot2)
subset <- sample (n, n/10)
n <- length (subset)
y <- y[subset]
X <- X[subset,]
state <- state[subset]
