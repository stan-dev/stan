## 
stan_model <- function(file, 
                       model_name = "anon_model", 
                       model_code = '', 
                       stanc_ret = NULL, 
                       boost_lib = NULL, 
                       eigen_lib = NULL, 
                       save_dso = TRUE,
                       verbose = FALSE, ...) { 

  # Construct a stan model from stan code 
  # 
  # Args: 
  #   file: the file that has the model in Stan model language.
  #   model_name: a character for naming the model. 
  #   stanc_ret: An alternative way to specify the model
  #     by using returned results from stanc. 
  #   model_code: if file is not specified, we can used 
  #     a character to specify the model.   

  if (is.null(stanc_ret)) {
    model_name2 <- deparse(substitute(model_code))
    if (is.null(attr(model_code, "model_name2")))
      attr(model_code, "model_name2") <- model_name2
    if (missing(model_name)) model_name <- NULL 
    stanc_ret <- stanc(file = file, model_code = model_code, 
                       model_name = model_name, verbose, ...)
  }
  if (!is.list(stanc_ret)) {
    stop("stanc_ret needs to be the returned object from stanc.")
  } 
  m <- match(c("cppcode", "model_name", "status"), names(stanc_ret)) 
  if (any(is.na(m))) {
    stop("stanc_ret does not have element `cppcode', `model_name', and `status'") 
  } else {
    if (!stanc_ret$status)
      stop("stanc_ret is not a successfully returned list from stanc")
  } 

  model_cppname <- stanc_ret$model_cppname 
  model_name <- stanc_ret$model_name 
  model_code <- stanc_ret$model_code 
  inc <- paste("#include <rstan/rstaninc.hpp>\n", 
               stanc_ret$cppcode, 
               get_Rcpp_module_def_code(model_cppname), 
               sep = '')  

  cat("COMPILING THE C++ CODE FOR MODEL '", model_name, "' NOW.\n", sep = '') 
  if (verbose) cat(system_info(), "\n")
  if (!is.null(boost_lib)) { 
    old.boost_lib <- rstan_options(boost_lib = boost_lib) 
    on.exit(rstan_options(boost_lib = old.boost_lib)) 
  } 

  if (!is.null(eigen_lib)) { 
    old.eigen_lib <- rstan_options(eigen_lib = eigen_lib) 
    on.exit(rstan_options(eigen_lib = old.eigen_lib)) 
  }

  
  dso <- cxxfunctionplus(signature(), body = paste(" return Rcpp::wrap(\"", model_name, "\");", sep = ''), 
                         includes = inc, plugin = "rstan", save_dso = save_dso,
                         module_name = paste('stan_fit4', model_cppname, '_mod', sep = ''), 
                         verbose = verbose) 
               
  obj <- new("stanmodel", model_name = model_name, 
             model_code = model_code, 
             dso = dso, # keep a reference to dso
             model_cpp = list(model_cppname = model_cppname, 
                              model_cppcode = stanc_ret$cppcode)) 
  invisible(obj) 
  ## We keep a reference to *dso* above to avoid dso to be 
  ## deleted by R's garbage collection. Note that if dso 
  ## is freed, we can lose the compiled shared object, which
  ## can cause segfault later. 
} 

is_sm_valid <- function(sm) {
  # Test if a stan model (compiled object) is still valid. 
  # It could become invalid when the user do not specify
  # save_dso when calling stan_model. So when the user
  # use the model created in another R session, the dso
  # is lost. 
  # 
  # Args:
  #   sm: the stanmodel object 
  # 
  if (is_dso_loaded(sm@dso)) return(TRUE)
  sm@dso@dso_saved && identical(sm@dso@system, R.version$system)
} 

##
##
## 

stan <- function(file, model_name = "anon_model", 
                 model_code = '', 
                 fit = NA, 
                 data = list(), 
                 pars = NA, 
                 chains = 4, iter = 2000, 
                 warmup = floor(iter / 2), 
                 thin = 1, 
                 init = "random", 
                 seed = sample.int(.Machine$integer.max, 1), 
                 sample_file, # the file to which the samples are written
                 save_dso = TRUE,
                 verbose = FALSE, ..., 
                 boost_lib = NULL, 
                 eigen_lib = NULL) {
  # Return a fitted model (stanfit object)  from a stan model, data, etc.  
  # A wrap of method stan_model and sampling of class stanmodel. 
  # 
  # Args:
  # 
  # Returns: 
  #   A S4 class stanfit object  

  if (is(fit, "stanfit")) sm <- get_stanmodel(fit)
  else { 
    attr(model_code, "model_name2") <- deparse(substitute(model_code))  
    if (missing(model_name)) model_name <- NULL 
    sm <- stan_model(file, model_name = model_name, model_code = model_code,
                     boost_lib = boost_lib, eigen_lib = eigen_lib, 
                     save_dso = save_dso, verbose = verbose, ...)
  }

  if (missing(sample_file))  sample_file <- NA 

  sampling(sm, data, pars, chains, iter, warmup, thin, seed, init, 
           sample_file = sample_file, verbose = verbose, check_data = TRUE, ...) 
} 
