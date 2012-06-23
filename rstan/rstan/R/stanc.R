stanc <- function(model.code, model.name = "anon_model", verbose = FALSE) {
  # Call stanc, which is written in C++
  # 
  SUCCESS_RC <- 0 
  EXCEPTION_RC <- -1
  PARSE_FAIL_RC <- -2 
  r <- .Call("stanc", model.code, model.name, PACKAGE = "rstan");
  if (is.null(r)) {
    stop(paste("Failed to run stanc for model '", model.name, 
               "' and no error message provided", sep = '')) 
  } else if (r$status == PARSE_FAIL_RC) {
    stop(paste("Failed to parse Stan model '", model.name, 
               "' and no error message provided"), sep = '') 
  } else if (r$status == EXCEPTION_RC) {
    stop(paste("Failed to parse Stan model '", model.name, 
               "' and error message provided as:\n", 
               r$msg, sep = '')) 
  } 

  if (r$status != SUCCESS_RC) {
    if (verbose)  
      cat("Successful of parsing the Stan model '", model.name, "'.\n") 
  } 
  r
}


stanc.version <- function() {
  .Call('stanc_version', PACKAGE = 'rstan')
}

