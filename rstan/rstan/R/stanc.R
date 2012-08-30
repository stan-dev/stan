# Part of the rstan package for an R interface to Stan 
# Copyright (C) 2012 Columbia University
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

stanc <- function(model_code, model_name = "anon_model", verbose = FALSE) {
  # Call stanc, which is written in C++
  # 

  cat("\nCOMPILING MODEL '", model_name, "' FROM Stan CODE TO C++ CODE NOW.\n", sep = '')
  SUCCESS_RC <- 0 
  EXCEPTION_RC <- -1
  PARSE_FAIL_RC <- -2 

  # model_name in C++, to avoid names that would be problematic in C++. 
  model_cppname <- legitimate_model_name(model_name) 
  r <- .Call("stanc", model_code, model_cppname, PACKAGE = "rstan")
  # from the cpp code of stanc,
  # returned is a named list with element 'status', 'model_cppname', and 'cppcode' 
  r$model_name <- model_name  
  r$model_code <- model_code 
  if (is.null(r)) {
    stop(paste("failed to run stanc for model '", model_name, 
               "' and no error message provided", sep = '')) 
  } else if (r$status == PARSE_FAIL_RC) {
    stop(paste("failed to parse Stan model '", model_name, 
               "' and no error message provided"), sep = '') 
  } else if (r$status == EXCEPTION_RC) {
    stop(paste("failed to parse Stan model '", model_name, 
               "' and error message provided as:\n", 
               r$msg, sep = '')) 
  } 

  if (r$status != SUCCESS_RC) {
    if (verbose)  
      cat("Successful of parsing the Stan model '", model_name, "'.\n") 
  } 
  r
}


stan_version <- function() {
  .Call('stanc_version', PACKAGE = 'rstan')
}

