library(rstan)
load("fxrates.rda")

standata <-
  within(list(), {
    y <- t(fxrates)
    r <- nrow(y)
    T <- ncol(y)
    m0 <- array(rep(0, r), r)
    C0 <- diag(3) * 1e7
  })


stan_rdump(names(standata), file="fx_equicorr.data.R", envir=as.environment(standata))
