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



## 
## 
stan.model <- function(file, 
                       model.name = "anon_model", 
                       model.code = '', 
                       stanc.ret = NULL, 
                       boost.lib = NULL, 
                       verbose = FALSE) { 

  # Construct a stan model from stan code 
  # 
  # Args: 
  #   file: the file that has the model in Stan model language.
  #   model.name: a character for naming the model. 
  #   stanc.ret: An alternative way to specify the model
  #     by using returned results from stanc. 
  #   model.code: if file is not specified, we can used 
  #     a character to specify the model.   

  if (is.null(stanc.ret)) {
    model.code <- get.model.code(file, model.code)  
    stanc.ret <- stanc(model.code, model.name) 
  } 
  if (!is.list(stanc.ret)) {
    stop("stanc.ret needs to be the returned object from stanc.")
  } 
  m <- match(c("cppcode", "model.name", "status"), names(stanc.ret)) 
  if (any(is.na(m))) {
    stop("stanc.ret does not have element `cppcode', `model.name', and `status'") 
  } else {
    if (stanc.ret$status != 0) 
      stop("stanc.ret is not a successfully returned list from stanc")
  } 

  model.cppname <- stanc.ret$model.cppname 
  model.name <- stanc.ret$model.name 
  model.code <- stanc.ret$model.code 
  inc <- paste("#include <rstan/rstaninc.hpp>\n", 
               stanc.ret$cppcode, 
               get_Rcpp_module_def_code(model.cppname), 
               sep = '')  

  cat("COMPILING THE C++ CODE FOR MODEL '", model.name, "' NOW.\n", sep = '') 
  if (!is.null(boost.lib)) { 
    old.boost.lib <- rstan.options(boost.lib = boost.lib) 
    tryCatch(fx <- cxxfunction(signature(), body = '  return R_NilValue;', 
                               includes = inc, plugin = "rstan", verbose = verbose),
             error = function(e) {rstan.options(boost.lib = old.boost.lib); stop(e)})
  } else {
    fx <- cxxfunction(signature(), body = '  return R_NilValue;', 
                      includes = inc, plugin = "rstan", verbose = verbose) 
  } 
               
  mod <- Module(model.cppname, getDynLib(fx)) 
  # stan_fit_cpp_module <- do.call("$", list(mod, model.name))
  stan_fit_cpp_module <- eval(call("$", mod, model.cppname))
  new("stanmodel", model.name = model.name, 
      model.code = model.code, 
      .modelmod = list(sampler = stan_fit_cpp_module, 
                       cxxfun = fx)) # keep a reference of fx

  ## We keep a reference to *fx* above to avoid fx to be 
  ## deleted by R's garbage collection. Note that if fx 
  ## is freed, we lost the compiled shared object, which
  ## could cause segfault later. 
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
                 fit = NA, 
                 data = list(), 
                 pars = NA, 
                 n.chains = 4L, n.iter = 2000L, 
                 n.warmup = floor(n.iter / 2), 
                 n.thin = 1L, 
                 init.t = "random", 
                 init.v = NULL, 
                 seed = sample.int(.Machine$integer.max, 1), 
                 sample.file, 
                 verbose = FALSE, ..., boost.lib = NULL) {
  # Return a fitted model (stanfit object)  from a stan model, data, etc.  
  # A wrap of method stan.model and sampling of class stanmodel. 
  # 
  # Args:
  # 
  # Returns: 
  #   A S4 class stanfit object  

  if (is(fit, "stanfit")) sm <- get.stanmodel(fit)
  else sm <- stan.model(file, verbose = verbose, model.name, model.code, boost.lib)

  if (missing(sample.file))  sample.file <- NA 

  # check data before compiling model, which typically takes more time 
  if (is.character(data)) data <- mklist(data) 
  if (!missing(data) && length(data) > 0) data <- data.preprocess(data)
  else data <- list()  

  sampling(sm, data, pars, n.chains, n.iter, n.warmup, n.thin, seed, init.t, init.v, 
           sample.file = sample.file, verbose = verbose, check.data = FALSE, ...) 
} 
