
/**
 * Part of the rstan package for an R interface to Stan 
 * Copyright (C) 2012 Columbia University
 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
**/
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
 * Tell if it is a NULL native symbol 
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

