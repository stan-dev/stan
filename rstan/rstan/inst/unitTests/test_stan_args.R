
require(rstan) 
inc <- paste(readLines('test_stan_args.cpp'), collapse = '\n')
fx <- cxxfunction(signature(x = "list"), 
                  body = "  return test_stan_args(x);", 
                  includes = inc, 
                  plugin = "rstan", verbose = TRUE)

a1 <- fx(list(iter = 100, thin = 100)) 
print(a1) 
