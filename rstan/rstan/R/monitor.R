# stan_prob_autocovariance <- function(v) { 
#   .Call("stan_prob_autocovariance", v, PACKAGE = 'rstan') 
# }

ess_rfun <- function(sims) {
  # Compute the effective sample size for samples of several chains 
  # for one parameter; see the C++ code of function  
  # effective_sample_size2 in chains.cpp 
  # 
  # Args:
  #   sims: a 2-d array _without_ warmup samples (# iter * # chains) 
  # 
  # Note: 
  #   The implementation in R uses acf in R to compute the autocovariance
  #   and the results might be a little bit different from that in stan. 
  #   The R function wrapping up the C++ implementation is defined in 
  #   chains.R with name rstan_ess2_cpp 
  if (is.vector(sims)) dim(sims) <- c(length(sims), 1)
  chains <- ncol(sims)
  n_samples <- nrow(sims)

  acov <- lapply(1:chains, 
                 FUN = function(i) {
                   cov <- acf(sims[,i], lag.max = n_samples - 1, 
                              plot = FALSE, type = c("covariance")) 
                   cov$acf[,,1]
                 }) 
#   acov <- lapply(1:chains, 
#                  FUN = function(i) stan_prob_autocovariance(sims[, i]))
  acov <- do.call(cbind, acov)
  chain_mean <- apply(sims, 2, mean)
  mean_var <- mean(acov[1,]) * n_samples / (n_samples - 1) 
  var_plus <- mean_var * (n_samples - 1) / n_samples
  if (chains > 1) 
    var_plus <- var_plus + var(chain_mean)
  rho_hat_sum <- 0
  for (t in 2:nrow(acov)) {
    rho_hat <- 1 - (mean_var - mean(acov[t, ])) / var_plus
    if (rho_hat < 0) break
    rho_hat_sum <- rho_hat_sum + rho_hat
  } 
  ess <- chains * n_samples
  if (rho_hat_sum > 0) ess <- ess / (1 + 2 * rho_hat_sum)
  ess 
} 

split_rhat_rfun <- function(sims) {
  # Compute the split rhat for the diagnostics of converging; 
  # see the C++ code of split_potential_scale_reduction in chains.cpp.  
  # 
  # Args:
  #   sims: a 2-d array _without_ warmup samples (# iter * # chains) 
  # 
  # Note: 
  #   The R function wrapping up the C++ implementation is defined 
  #   in chains.R with name rstan_splitrhat2_cpp 
  if (is.vector(sims)) dim(sims) <- c(length(sims), 1)
  chains <- ncol(sims)
  n_samples <- nrow(sims)
  half_n <- floor(n_samples / 2)
  # cat("n_samples=", n_samples, "\n"); cat("chains=", chains, "\n")
  # cat("half_n=", half_n, "\n")
  idx_2nd <- n_samples - half_n + 1
  
  split_chain_mean <- numeric(chains * 2)
  split_chain_var <- numeric(chains * 2)
  
  for (i in 1:chains) {
    split_chain_mean[i] <- mean(sims[1:half_n, i])
    split_chain_var[i] <- var(sims[1:half_n, i])
    split_chain_mean[chains + i] <- mean(sims[idx_2nd:n_samples, i])
    split_chain_var[chains + i] <- var(sims[idx_2nd:n_samples, i])
  } 
  var_between <- half_n * var(split_chain_mean)
  var_within <- mean(split_chain_var) 
  sqrt((var_between/var_within + half_n -1)/half_n)
} 

monitor <- function(sims, warmup = floor(dim(sims)[1] / 2), 
                    probs = c(0.025, 0.25, 0.50, 0.75, 0.975), 
                    digits_summary = 1, 
                    print = TRUE, ...) { 
  # print the summary for a general simulation results 
  # of 3-d array: # iter * # chains * # parameters 
  # Args:
  #   sims: a 3-d array described above 
  #   warmup: the number of iterations used for warmup 
  #   probs: probs of summarizing quantiles 
  #   print: print out the results
  # 
  # Return: 
  #   A summary array  
  dim_sims <- dim(sims)
  dimnames_sims <- dimnames(sims)
  parnames <- dimnames_sims[[3]]
  if (length(dim_sims) != 3) 
    stop("'sims' is not a 3-d array")
  if (warmup > dim_sims[1])
    stop("warmup is larger than the total number of iterations")
  num_par <- dim_sims[3]
  
  if (is.null(parnames)) parnames <- paste0("V", 1:num_par)
  sims_wow <- if (warmup >= 1) apply(sims, c(2, 3), FUN = function(x) x[-(1:warmup)]) else sims 
  m <- apply(sims_wow, 3, mean)
  sd <- sapply(1:num_par, FUN = function(i) sd(as.vector(sims_wow[,,i]))) 
  quan <- lapply(1:num_par, FUN = function(i) quantile(sims_wow[,,i], probs = probs))
  probs_str <- names(quan[[1]])
  quan <- do.call(rbind, quan)
  rhat <- sapply(1:num_par, FUN = function(i) split_rhat_rfun(sims_wow[,,i]))
  ess <- sapply(1:num_par, FUN = function(i) ess_rfun(sims_wow[,,i]))
  sem <- sd / sqrt(ess)
 
  summary <- cbind(m, sem, sd, quan, ess, rhat)
  colnames(summary) <- c("mean", "se_mean", "sd", probs_str, "n_eff", "Rhat")
  rownames(summary) <- parnames 
  if (print) {
    cat("Inference for the input samples (")
    cat(dim_sims[2], " chains: each with iter=", dim_sims[1], "; warmup=", warmup, "):\n\n", sep = "")
    # round n_eff to integers
    summary[, 'n_eff'] <- round(summary[, 'n_eff'], 0)
    print(round(summary, digits_summary), ...)
 
    cat("\nFor each parameter, n_eff is a crude measure of effective sample size,\n", 
        "and Rhat is the potential scale reduction factor on split chains (at \n",
        "convergence, Rhat=1).\n", sep = '')
  } 
  invisible(summary) 
} 

