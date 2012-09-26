
.setup <- function() { 
  require(rstan) 
} 

test_stan_args <- function() { 
  inc <- paste(readLines('test_stan_args.cpp'), collapse = '\n')
  fx <- cxxfunction(signature(x = "list"), 
                    body = "  return test_stan_args(x);", 
                    includes = inc, 
                    plugin = "rstan", verbose = TRUE)

  a1 <- fx(list(iter = 100)) 
  a2 <- fx(list(iter = 100, thin = 3, refresh = 1)) 
  a3 <- fx(list(iter = 5, thin = 3, refresh = -1)) 
  checkEquals(a1$iter, 100) 
  checkEquals(a1$thin, 1) 
  checkEquals(a2$iter, 100) 
  checkEquals(a2$thin, 3) 
  checkEquals(a3$iter, 5) 
  checkEquals(a3$refresh, -1) 
} 
