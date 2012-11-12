
library(rstan) 


inc <- '
#include <iostream>
#include <Rcpp.h>
#include <R.h>
#include <Rinternals.h>

#include <rstan/io/r_ostream.hpp> 

RcppExport SEXP test_r_ostream(SEXP params) {
  std::string str = Rcpp::as<std::string>(params);
  rstan::io::rcout << "[rcout] " << str << std::endl; 
  rstan::io::rcerr << "[rcerr] " << str; // << std::endl;
  rstan::io::rcerr.flush();
  rstan::io::rcerr << std::endl;
  return Rcpp::wrap(true);
} 

RCPP_MODULE(rstantest){
  using namespace Rcpp;
  function("test_r_ostream", &test_r_ostream); 
}
' 

fx <- cxxfunction(signature(), body = 'return R_NilValue;\n',
                  includes = inc,
                  plugin = "rstan", verbose = TRUE) 

mod <- Module("rstantest", getDynLib(fx)) 
mod$test_r_ostream('hello1')
mod$test_r_ostream('hello2')
mod$test_r_ostream('hello3')
mod$test_r_ostream('hello4')
mod$test_r_ostream('hello5')

