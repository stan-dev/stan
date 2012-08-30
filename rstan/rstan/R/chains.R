rstan_ess <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  ess <- .Call("effective_sample_size", sim, n - 1, PACKAGE = "rstan")
  ess
} 

rstan_splitrhat <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  rhat <- .Call("split_potential_scale_reduction", sim, n - 1, PACKAGE = "rstan")
  rhat
} 
