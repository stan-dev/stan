
#include <rstan/stan_args.hpp> 
// #include <rstan/rstaninc.hpp>

RcppExport SEXP test_stan_args(SEXP in) {
  BEGIN_RCPP 
  Rcpp::List lst(in); 
  rstan::stan_args args(lst); 
  return args.stan_args_to_rlist(); 
  END_RCPP 
} 
