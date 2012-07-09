
// #include <Rcpp.h>
// #include <string>
// #include <iostream>
#include <stan/version.hpp>
#include <stan/gm/compiler.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Rcpp.h>

RcppExport SEXP stanc(SEXP model_stancode, SEXP model_name);
RcppExport SEXP stanc_version(); 

SEXP stanc_version() {
  BEGIN_RCPP;
  std::string stan_version 
    = stan::MAJOR_VERSION + "." +
      stan::MINOR_VERSION + "." +
      stan::PATCH_VERSION;
  return  Rcpp::wrap(stan_version); 
  END_RCPP;
} 

SEXP stanc(SEXP model_stancode, SEXP model_name) { 
  BEGIN_RCPP;
  static const int SUCCESS_RC = 0;
  static const int EXCEPTION_RC = -1;
  static const int PARSE_FAIL_RC = -2;
  
  static const bool INCLUDE_MAIN = true; 
  /*
  std::string stan_version 
    = stan::MAJOR_VERSION + "." +
      stan::MINOR_VERSION + "." +
      stan::PATCH_VERSION;
  */

  std::string mcode_ = Rcpp::as<std::string>(model_stancode); 
  std::string mname_ = Rcpp::as<std::string>(model_name); 
   
  std::stringstream out;
  std::istringstream in(mcode_); 
  try {
    bool valid_model
      = stan::gm::compile(in,out,mname_,!INCLUDE_MAIN);
    if (!valid_model) {
      return Rcpp::List::create(Rcpp::Named("status") = PARSE_FAIL_RC); 

    }
  } catch(const std::exception& e) {
    // REprintf("\nERROR PARSING\n %s\n", e.what()); 
    return Rcpp::List::create(Rcpp::Named("status") = EXCEPTION_RC,
                              Rcpp::Named("msg") = Rcpp::wrap(e.what())); 
  }
  return Rcpp::List::create(Rcpp::Named("status") = SUCCESS_RC, 
                            Rcpp::Named("model_name") = mname_,
                            Rcpp::Named("cppcode") = out.str());

  END_RCPP;
}


