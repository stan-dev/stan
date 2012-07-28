stanc <- function(model.code, model.name = "anon_model", verbose = FALSE) {
  # Call stanc, which is written in C++
  # 
  SUCCESS_RC <- 0 
  EXCEPTION_RC <- -1
  PARSE_FAIL_RC <- -2 
  model.name2 <- legitimate.model.name(model.name) 
  r <- .Call("stanc", model.code, model.name2, PACKAGE = "rstan")
  # from the cpp code of stanc,
  # returned is a named list with element 'status', 'model.name', and 'cppcode' 
  r$model.name2 <- model.name2  
  r$model.code <- model.code 
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

