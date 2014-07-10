library(rstan)
load("fxrates.rda")

standata <-
  within(list(), {
    y <- t(fxrates)
    r <- nrow(y)
    T <- ncol(y)
    m0 <- array(0, 1)
    C0 <- matrix(1e7)
  })


stan_rdump(names(standata), file="fx_factor.data.R", envir=as.environment(standata))
