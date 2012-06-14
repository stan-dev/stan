
## 
## @param stan.home TBD 
## 
stan.model <- function(file, verbose = FALSE, 
                       model.name = "anon_model", 
                       model.code = '', stan.home) {

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
      .modelmod = list(sampler = stan_fit_cpp_module)) 

} 

##
##
## 

stan <- function(file, model.name = "anon_model", 
                 model.code = '', 
                 data, n.chains = 1L, n.iter = 2000L, 
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
