
#ifndef __RSTAN__IO__RLIST_VAR_CONTEXT_HPP__
#define __RSTAN__IO__RLIST_VAR_CONTEXT_HPP__

#include <cstddef>
#include <stdexcept>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <iostream>
#include <locale>

#include <boost/throw_exception.hpp>

#include <stan/math/matrix.hpp>
#include <stan/io/var_context.hpp>

#include <Rcpp.h>

namespace rstan {

  namespace io {

    /**
    namespace {
       size_t product(std::vector<size_t> dims) {
         size_t y = 1U;
         for (size_t i = 0; i < dims.size(); ++i)
           y *= dims[i];
         return y;
       }
    }
    **/

    /**
     * Represents named arrays with dimensions.
     *
     * A rlist_var_context implements var_context from
     * a R list (Rcpp::List) --- named variables (scalar,
     * vector, array) with dimensions.  The values for
     * an array are typed to doulbe or int.  However,
     * it is R's job to pass data with correct types
     * though R uses double as the default atomic type. 
     * 
     * <p>See <code>dump</code> for more information as 
     * rlist_var_context only differs from that 
     * by the constructor. 
     *
     * <p>The dimensions and values of variables
     * may be accessed by name. 
     */
    class rlist_var_context : public stan::io::var_context {
    private: 
      std::map<std::string, 
               std::pair<std::vector<double>,
                         std::vector<size_t> > > vars_r_;
      std::map<std::string, 
               std::pair<std::vector<int>, 
                         std::vector<size_t> > > vars_i_;
      std::vector<double> const empty_vec_r_;
      std::vector<int> const empty_vec_i_;
      std::vector<size_t> const empty_vec_ui_;
      /**
       * Return <code>true</code> if this rlist_var_context contains the
       * specified variable name is defined in the real values. This method
       * returns <code>false</code> if the values are all integers.
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable exists in the 
       * real values of the rlist_var_context.
       */
      bool contains_r_only(const std::string& name) const {
        return vars_r_.find(name) != vars_r_.end();
      }
    public: 
      rlist_var_context() {} 

      /**
       * Construct a rlist_var_context object from the specified two R list.
       *
       * @param in Input of R list (represented by Rcpp::List) 
       * from which to read.
       */
      rlist_var_context(Rcpp::List &in) {
        // should I use iterators? 
        std::vector<std::string> varnames 
          = Rcpp::as<std::vector<std::string> >(in.names()); 
        // Rprintf("in.size()=%d.\n", in.size()); 
        for (size_t i = 0; i < in.size(); i++) {
          SEXP ee = in[i]; 
          SEXP dim = Rf_getAttrib(ee, R_DimSymbol); 
          R_len_t eelen = Rf_length(ee); 
          if (eelen < 1) continue; 

          // Note that in R, the default is real, which causes problems as we
          // are thinking they are integers, but Rf_isInteger would return
          // FALSE. One solution is use L suffix. Anyway, the R code should
          // change the data to integers when it is convertible. 
          if (Rf_isInteger(ee)) {
            if (Rf_length(dim) > 0) { 
              vars_i_[varnames[i]] 
                = std::pair<std::vector<int>, std::vector<size_t> >(
                  Rcpp::as<std::vector<int> >(ee), 
                  Rcpp::as<std::vector<size_t> >(dim)); 
            } else {
              if (1 == eelen) {  // scalar 
                vars_i_[varnames[i]] 
                  = std::pair<std::vector<int>, std::vector<size_t> >(
                    Rcpp::as<std::vector<int> >(ee), empty_vec_ui_); 
              } else { // vector 
                vars_i_[varnames[i]] 
                  = std::pair<std::vector<int>, std::vector<size_t> >(
                    Rcpp::as<std::vector<int> >(ee), std::vector<size_t>(1, eelen));
              }
            } 
          } else if(Rf_isNumeric(ee)) {
            if (Rf_length(dim) > 0) { 
              vars_r_[varnames[i]] 
                = std::pair<std::vector<double>, std::vector<size_t> >(
                  Rcpp::as<std::vector<double> >(ee), 
                  Rcpp::as<std::vector<size_t> >(dim)); 
            } else {
              if (1 == eelen) { 
                vars_r_[varnames[i]] 
                  = std::pair<std::vector<double>, std::vector<size_t> >(
                    Rcpp::as<std::vector<double> >(ee), empty_vec_ui_); 
              } else {
                vars_r_[varnames[i]] 
                  = std::pair<std::vector<double>, std::vector<size_t> >(
                    Rcpp::as<std::vector<double> >(ee), std::vector<size_t>(1, eelen));
              }
            }
          } else {
            continue; // ignore non-numeric data 
            // or should we report error? 
          } 
        } 
      } 

      /**
       * Return <code>true</code> if this rlist_var_context contains the
       * specified variable name is defined. This method returns
       * <code>true</code> even if the values are all integers.
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable exists.
       */
      bool contains_r(const std::string& name) const {
        return contains_r_only(name) || contains_i(name);
      }

      /**
       * Return <code>true</code> if this rlist_var_context contains an integer
       * valued array with the specified name. 
       *
       * @param name Variable name to test.
       * @return <code>true</code> if the variable name has an integer
       * array value.
       */
      bool contains_i(const std::string& name) const {
        return vars_i_.find(name) != vars_i_.end();
      }

      /**
       * Return the double values for the variable with the specified
       * name or null. 
       *
       * @param name Name of variable.
       * @return Values of variable.
       */
      std::vector<double> vals_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).first;
        } else if (contains_i(name)) {
          std::vector<int> vec_int = (vars_i_.find(name)->second).first;
          std::vector<double> vec_r(vec_int.size());
          for (size_t ii = 0; ii < vec_int.size(); ii++) {
            vec_r[ii] = vec_int[ii];
          }
          return vec_r;
        }
        return empty_vec_r_;
      }
      
      /**
       * Return the dimensions for the double variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Dimensions of variable.
       */
      std::vector<size_t> dims_r(const std::string& name) const {
        if (contains_r_only(name)) {
          return (vars_r_.find(name)->second).second;
        } else if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }

      /**
       * Return the integer values for the variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Values.
       */
      std::vector<int> vals_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).first;
        }
        return empty_vec_i_;
      }
      
      /**
       * Return the dimensions for the integer variable with the specified
       * name.
       *
       * @param name Name of variable.
       * @return Dimensions of variable.
       */
      std::vector<size_t> dims_i(const std::string& name) const {
        if (contains_i(name)) {
          return (vars_i_.find(name)->second).second;
        }
        return empty_vec_ui_;
      }

      /**
       * Return a list of the names of the floating point variables in
       * the rlist_var_context.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_r(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<double>,
                                std::vector<size_t> > >
                 ::const_iterator it = vars_r_.begin();
             it != vars_r_.end(); ++it)
          names.push_back((*it).first);
      }

      /**
       * Return a list of the names of the integer variables in
       * the rlist_var_context.
       *
       * @param names Vector to store the list of names in.
       */
      virtual void names_i(std::vector<std::string>& names) const {
        names.resize(0);        
        for (std::map<std::string, 
                      std::pair<std::vector<int>, 
                                std::vector<size_t> > >
                 ::const_iterator it = vars_i_.begin();
             it != vars_i_.end(); ++it)
          names.push_back((*it).first);
      }

      bool remove(const std::string& name) {
        return (vars_i_.erase(name) > 0) 
          || (vars_r_.erase(name) > 0);
      }
      
    };
    

  }


}


#endif

