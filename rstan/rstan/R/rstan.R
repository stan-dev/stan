
## 
## @param stan.home TBD 
## 
stan.model <- function(file, verbose = FALSE, 
                       model.name = "anon_model", 
                       model.code = '', stan.home) {

  # Construct a stan model from stan code 
  # 
  # Args: 
  #   file: the file that has the model in Stan model language.
  #   model.name: a character for naming the model. Note that
  #     the name needs to be a valid C++ name. So 
  #   model.code: if file is not specified, we can used 
  #     a character to specify the model.   
  #   stan.home: not used now, it mighted be needed
  #     if Rstan is packaged differently. 

  model.code <- get.model.code(file, model.code)  
  r <- stanc(model.code, model.name); 
  inc <- paste("#include <rstan/rstaninc.hpp>\n", 
               r$cppcode, 
               get_Rcpp_module_def_code(model.name), 
               sep = ''); 
  
  fx <- cxxfunction(signature(), body = '  return R_NilValue;', 
                    includes = inc, plugin = "rstan", verbose = verbose) 
               

  mod <- Module(model.name, getDynLib(fx)) 
  # stan_fit_cpp_module <- do.call("$", list(mod, model.name))
  stan_fit_cpp_module <- eval(call("$", mod, model.name))
  new("stanmodel", model.name = model.name, 
      .modelmod = list(sampler = stan_fit_cpp_module, 
                       cxxfun = fx)) # keep a reference of fx

  ## We keep a reference to *fx* above to avoid fx to be 
  ## deleted by R's garbage collection. Note that if fx 
  ## is freed, we lost the compiled shared object, which
  ## then cause segfault later. 

} 

is.sm.valid <- function(sm) {
  # Test if a stan model (compiled object) is still valid. 
  # It could become invalid when the user for example 
  # save this object and then load it in another R session
  # because the compiled model is lost. 
  # 
  # Args:
  #   sm: the stanmodel object 
  # Note:  
  # This depends on currently that we return R_NilValue
  # in the `src` when calling cxxfunction. 
  # 
  fx <- sm@.modelmod$cxxfun 
  r <- tryCatch(fx(), error = function(e) FALSE)
  if (is.null(r)) return(TRUE) 
  FALSE
} 

##
##
## 

stan <- function(file, model.name = "anon_model", 
                 model.code = '', 
                 data = list(), n.chains = 1L, n.iter = 2000L, 
                 n.warmup = floor(n.iter / 2), 
                 n.thin = 1L, 
                 init.t = "random", 
                 init.v = NULL, 
                 seed, 
                 sample.file, 
                 stan.home, 
                 verbose = FALSE, ...) {
  # Return a fitted model (stanfit object)  from a stan model, data, etc.  
  # A wrap of method stan.model and sampling of class stanmodel. 
  # 
  # Args:
  # 
  # Returns: 
  #   A S4 class stanfit object  
  
  sm <- stan.model(file, verbose = verbose, model.name, model.code, stan.home)
  sampling(sm, data, n.chains, n.iter, n.warmup, n.thin, init.t, init.v, seed, 
           sample.file, verbose = verbose) 
} 
