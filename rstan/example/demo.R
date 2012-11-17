
stopifnot(require(rstan))
stopifnot(require(parallel))

example(stan_demo)

bench <- function(i) {
  mark <- "elapsed"
  compile <- system.time(fit <- stan_demo(i, chains = 0, iter = 0))[mark]
  execute <- system.time(fit <- stan_demo(i, fit = fit, seed = 12345, refresh = -1))[mark]
  if (length(dim(fit)) == 0) execute <- NA_real_ # failure
  return(c(compile, execute))
}

results <- mclapply(1:length(model_names), mc.cores = 4L, FUN = bench,
                    mc.preschedule = FALSE)

names(results) <- model_names
save(results, file = 'time1.RData')
