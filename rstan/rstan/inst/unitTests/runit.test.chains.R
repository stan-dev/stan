
#   ess.fun <- function(sim, n) {
#     ess <- .Call("effective_sample_size", sim, n, PACKAGE = "rstan");
#     ess 
#   } 

#   rhat.fun <- function(sim, n) {
#     rhat <- .Call("split_potential_scale_reduction", sim, n, PACKAGE = "rstan");
#     rhat
#   } 

.setUp <- function() {
  require(rstan)
} 

test.essnrhat <- function() {
  upath <- system.file('unitTests', package='rstan')
  f1 <- file.path(upath, 'testdata', 'blocker1.csv')  
  f2 <- file.path(upath, 'testdata', 'blocker2.csv')  
  c1 <- read.csv(f1, comment.char = "#", header = TRUE)[, -(1:2)] 
  # c1 <- do.call(cbind, c1)  
  c2 <- read.csv(f2, comment.char = "#", header = TRUE)[, -(1:2)]
  # c2 <- do.call(cbind, c2)  
  lst <- list(samples = list(c1 = c1, c2 = c2), 
              n.save = c(nrow(c1), nrow(c2)), 
              permutation = NULL, 
              n.warmup2 = rep(0, 2), n.chains = 2, n.flatnames = ncol(c1))
  ess <- rstan:::rstan.ess(lst, 3)
  # cat("ess=", ess, "\n") 
  checkEquals(ess, 13.0778, tolerance = 0.001); 
  rhat <- rstan:::rstan.splitrhat(lst, 3)
  # cat("rhat=", rhat, "\n") 
  checkEquals(rhat, 1.187, tolerance = 0.001); 
  ess2 <- rstan:::rstan.ess(lst, 46)
  # cat("ess=", ess2, "\n") 
  checkEquals(ess2, 43.0242, tolerance = 0.001); 
  rhat2 <- rstan:::rstan.splitrhat(lst, 46) 
  # cat("rhat=", rhat2, "\n") 
  checkEquals(rhat2, 1.03715, tolerance = 0.001); 
} 






