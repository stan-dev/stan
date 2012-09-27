# to test part of stan_args.hpp
require(rstan) 
src <- ' 
  Rcpp::List lst(x); 
  rstan::stan_args args(lst); 
  return args.stan_args_to_rlist(); 
' 
fx <- cxxfunction(signature(x = "list"), 
                  body = src, 
                  includes = "#include <rstan/stan_args.hpp>", 
                  plugin = "rstan", verbose = TRUE)

a1 <- fx(list(iter = 100, thin = 100)) 
a2 <- fx(list(iter = 100, thin = 3)) 
a3 <- fx(list(iter = 5, thin = 3, refresh = -1)) 
a4 <- fx(list(iter = 5, thin = 3, refresh = -1, test_grad = TRUE)) 
print(a1) 
checkEquals <- function(x, y) {
  if (x != y)  stop(x, " != ",  y)
} 

checkEquals(a1$iter, 100) 
checkEquals(a1$thin, 100) 
checkEquals(a2$iter, 100) 
checkEquals(a2$thin, 3) 
checkEquals(a3$iter, 5) 
checkEquals(a3$test_grad, FALSE)
checkEquals(a3$refresh, -1) 
checkEquals(a4$test_grad, TRUE)

