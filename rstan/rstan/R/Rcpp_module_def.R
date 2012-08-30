get_Rcpp_module_def_code <- function(model_name) {
  def_Rcpp_module_hpp_file <- 
    system.file('include', '/rstan/rcpp_module_def_for_rstan.hpp', package = 'rstan') 
  if (def_Rcpp_module_hpp_file == '') 
    stop("Rcpp module definition file for rstan is not found.\n") 
  src <- paste(readLines(def_Rcpp_module_hpp_file), collapse = '\n')
  gsub("%model_name%", model_name, src)
} 

