
.setup <- function() { 
  require(rstan) 
} 

test_stan_args <- function() { 
  src <- ' 
    Rcpp::List lst(x); 
    rstan::stan_args args(lst); 
    return args.stan_args_to_rlist(); 
  ' 
  fx <- cxxfunction(signature(x = "list"), 
                    body = src, 
                    includes = "#include <rstan/stan_args.hpp>", 
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
