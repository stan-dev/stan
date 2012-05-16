
#ifndef __RSTAN__RSTAN_HPP__
#define __RSTAN__RSTAN_HPP__
#include <stan/model/model_header.hpp>
#include <rlist_var_context.hpp> 

namespace rstan {

  class rstan {
  private:
    rlist_var_context rlist_; 
  public: 
    rstan() {} 
    bool init(Rcpp::List in, Rcpp::List conf) {
      return Rcpp::wrap<true>; 
    } 
    void nuts_command() {
    } 
    
  };
} 

#endif 
