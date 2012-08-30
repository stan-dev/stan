// #include <stan/mcmc/chains.hpp> 

#include <R.h>
#include <Rinternals.h>

#ifdef __cplusplus
extern "C" {
#endif

extern SEXP is_Null_NS(SEXP ns); 

#ifdef __cplusplus
}
#endif


/*
 * Tell if it is a NULL native symbol. 
 * This function mainly used to tell if a function created by cxxfunction of R
 * package inline points to a NULL address, which would happen when it is
 * deserialized (that is, loaded from what is saved previously). 
 *  
 */
SEXP is_Null_NS(SEXP ns) {
  SEXP ans;
  PROTECT(ans = allocVector(LGLSXP, 1));
  LOGICAL(ans)[0] = 1;
  PROTECT(ns);
  if (TYPEOF(ns) == EXTPTRSXP) {
    // Rprintf("ptr=%p.\n", EXTPTR_PTR(ns));
    if (EXTPTR_PTR(ns) != NULL) LOGICAL(ans)[0] = 0;
  }
  UNPROTECT(2);
  return ans;
}

