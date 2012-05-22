#ifndef  __STANC_HPP__ 
#define  __STANC_HPP__ 

#include <Rcpp.h>
RcppExport SEXP stanc(SEXP model_stancode, SEXP model_name);
RcppExport SEXP stanc_version(); 
#endif 

