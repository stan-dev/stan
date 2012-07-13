#ifndef __RSTAN__RCPP_PATCH_HPP__
#define __RSTAN__RCPP_PATCH_HPP__

/**
 * Rcpp has problem with 64 bits integer, so the following could be 
 * used to downsize some 64 bits size_t used in Stan.  
 * So in the rstan code, we avoid Rcpp::wrap/as for size_t. So the 
 * following code could be used to find unnoticed using of 
 * Rcpp::wrap/as for size_t. 
 */ 

#if 0
#include <Rcpp.h>
#include <rstan/io/r_ostream.hpp>

namespace Rcpp {
  template <>  SEXP wrap(const std::vector<size_t>& nv) {
    Rcpp::IntegerVector v(nv.size());
    rstan::io::rcout << "wrap:";
    for (int i = 0; i < nv.size(); i++) {
      v[i] = nv[i];
      rstan::io::rcout << v[i] << ", ";
    }
    rstan::io::rcout << std::endl;
    return v;
  }

  template <>  SEXP wrap(const size_t& n) {
    Rcpp::IntegerVector v(1); 
    v[0] = n;
    rstan::io::rcout << "n=" << n << std::endl;
    return v;
  } 
  
  template <> size_t as(SEXP s) {
    int i = Rcpp::as<int>(s);
    rstan::io::rcout << "i=" << i << std::endl;
    return i;
  } 

  template <> std::vector<size_t> as (SEXP s) {
    Rcpp::IntegerVector v1(s);
    std::vector<size_t> v2;
    rstan::io::rcout << "as";
    for (int i = 0; i < v1.size(); i++) {
      v2.push_back(v1[i]);
      rstan::io::rcout << "v1[i]" << ", ";
    }
    rstan::io::rcout << std::endl;
    return v2;
  }
}
#endif 

#endif 
