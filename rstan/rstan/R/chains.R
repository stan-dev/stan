rstan_ess <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  ess <- .Call("effective_sample_size", sim, n - 1, PACKAGE = "rstan")
  ess
} 

rstan_splitrhat <- function(sim, n) {
  # Args:
  #   n: Chain index starting from 1.
  if (sim$n_save[1] - sim$warmup2[1] < 2) return(NaN)
  rhat <- .Call("split_potential_scale_reduction", sim, n - 1, PACKAGE = "rstan")
  rhat
}

rstan_seq_perm <- function(n, chains, seed, chain_id = 1) {
  # Args:
  #   n: length of sequence to be generated 
  #   chains: the number of chains, for which the permuations are applied
  #   seed: the seed for RNG 
  #   chain_id: the chain id, for which the returned permuation is applied 
  # 
  conf <- list(n = n, chains = chains, seed = seed, chain_id = chain_id) 
  perm <- .Call("seq_permutation", conf, PACKAGE = 'rstan')
  perm + 1L # start from 1 
} 
