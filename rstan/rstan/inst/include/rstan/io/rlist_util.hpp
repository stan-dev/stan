#ifndef __RSTAN__IO__RLIST_UTIL_HPP__ 
#define __RSTAN__IO__RLIST_UTIL_HPP__ 

namespace rstan {

  /** 
   * <p> Get R list element by name. Return NULL is there is no element 
   * in the list with the name. 
   * 
   * @param lst The <code>Rcpp::List</code> from R's list. 
   * @param name The name of the element in the list that we are interested. 
   * @return The element in form of R's SEXP. If there is not a element 
   * with the name of our interests, return NULL (R_NilValue). 
   */ 
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
