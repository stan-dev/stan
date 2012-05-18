#ifndef __RSTAN_RLIST_UTIL_HPP__ 
#define __RSTAN_RLIST_UTIL_HPP__ 

namespace rstan {

  SEXP get_list_element_by_name(Rcpp::List lst, const std::string& name) {
    try {
      return lst[name]; 
    } catch (Rcpp::index_out_of_bounds &e) {
      return R_NilValue; 
    } catch (...) { 
      throw; 
    } 
  } 

} 

#endif 
