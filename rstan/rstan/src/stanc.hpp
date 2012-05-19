#ifndef  _myrstanpkg_STAN2CPP_H
#define  _myrstanpkg_STAN2CPP_H

#include <Rcpp.h>
RcppExport SEXP stanc(SEXP model_stancode, SEXP model_name);
RcppExport SEXP version(); 
#endif 

