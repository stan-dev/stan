stanc <- function(file, model_code = '', model_name = "anon_model", verbose = FALSE) {
  # Call stanc, which is written in C++
  # 
  model_name2 <- deparse(substitute(model_code))  
  if (is.null(attr(model_code, "model_name2"))) 
    attr(model_code, "model_name2") <- model_name2 
  model_code <- get_model_strcode(file, model_code)  
  if (missing(model_name) || is.null(model_name)) 
    model_name <- attr(model_code, "model_name2") 
  cat("\nTRANSLATING MODEL '", model_name, "' FROM Stan CODE TO C++ CODE NOW.\n", sep = '')
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
  
  r$status = if (r$status == 0) TRUE else FALSE

  if (r$status != SUCCESS_RC) {
    if (verbose)  
      cat("successful of parsing the Stan model '", model_name, "'.\n", sep = '') 
  } 
  invisible(r)
}


stan_version <- function() {
  .Call('stan_version', PACKAGE = 'rstan')
}

