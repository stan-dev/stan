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


setMethod("show", "stanmodel",
          function(object) {
            cat("S4 class stanmodel: `", object@model.name, "' coded as follows:\n" ,sep = '') 
            cat(object@model.code, "\n")
          }) 

#   setMethod("plot", "stanmodel",
#             function(x, y, ...) {
#               cat("plot method of class stanmodel.\n") 
#             }) 

#   setMethod("print", "stanmodel",
#             function(x, ...) {
#               cat("print method of class stanmodel.\n") 
#             }) 

## add extract to the list of the methods that R knows.
#   setGeneric(name = "extract",
#              def = function(object, x) { standardGeneric("extract")})

#   setMethod("extract", "stanmodel",
#             function(object, x) {
#               cat("intend to return samples for parameters x.\n")
#             }) 


setGeneric(name = "sampling",
           def = function(object, ...) { standardGeneric("sampling")})


# data, n.chains = 1L, n.iter = 2000L,
#                         n.warmup = floor(n.iter / 2),
#                         n.thin = 1L, init.t = 'random', init.v = NULL, pars, seed,
#                         sample.file, ...,
#                         verbose = FALSE) { standardGeneric("sampling")})

setMethod("sampling", "stanmodel",
          function(object, data = list(), pars = NA, n.chains = 4L, n.iter = 2000L,
                   n.warmup = floor(n.iter / 2),
                   n.thin = 1L, seed = sample.int(.Machine$integer.max, 1),
                   init.t = "random", init.v = NULL, check.data = TRUE, 
                   sample.file, verbose = FALSE, ...) {

            if (!is.sm.valid(object))
              stop("the compiled model from C++ code is not valid any more")

            if (!is.dso.loaded(object@dso)) {
               grab.cxxfun(object@dso) 
               model.cppname <- object@.modelmod$model.cppname  
               mod <- Module(model.cppname, getDynLib(object@dso)) 
               stan_fit_cpp_module <- eval(call("$", mod, model.cppname))
               assign("sampler", stan_fit_cpp_module, envir = object@.modelmod)
            }

            if (n.chains < 1) 
              stop("The number of chains (n.chains) is less than 1")

            if (check.data) { 
              # allow data to be specified as a vector of character string 
              if (is.character(data)) data <- mklist(data) 

              # check data and preprocess
              if (!missing(data) && length(data) > 0) data <- data.preprocess(data)
              else data <- list()
            } 

            sampler <- new(object@.modelmod$sampler, data)
            m.pars = sampler$param_names() 
            p.dims = sampler$param_dims() 
            if (!missing(pars) && !is.na(pars) && length(pars) > 0) {
              sampler$update_param_oi(pars)
              m <- which(match(pars, m.pars, nomatch = 0) == 0)
              if (length(m) > 0) 
                stop("No parameter ", paste(pars[m], collapse = ', ')) 
            }

            args.list <- config.argss(n.chains = n.chains, n.iter = n.iter,
                                      n.warmup = n.warmup, n.thin = n.thin,
                                      init.t = init.t, init.v = init.v,
                                      seed = seed, sample.file, ...)
            n.save <- 1 + (n.iter - 1) %/% n.thin 
            # number of samples saved after thinning
            n.warmup2 <- 1 + (n.warmup - 1) %/% n.thin 
            n.kept <- n.save - n.warmup2 
            samples <- vector("list", n.chains)

            for (i in 1:n.chains) {
              # cat("[sampling:] i=", i, "\n")
              # print(args.list[[i]])
              cat("SAMPLING FOR MODEL '", object@model.name, "' NOW (CHAIN ", i, ").\n", sep = '')
              samples[[i]] <- sampler$call_sampler(args.list[[i]])
              if (is.null(samples[[i]])) 
                stop("Error occurred when calling the sampler")
            }

            permutation.lst <-
              lapply(1:n.chains, function(id) sampler$permutation(c(n.kept, 1, id))) 

            fnames.oi <- sampler$param_fnames_oi()
            n.flatnames <- length(fnames.oi)
            sim = list(samples = samples,
                       n.iter = n.iter, n.thin = n.thin, 
                       n.warmup = n.warmup, 
                       n.chains = n.chains,
                       n.save = rep(n.save, n.chains),
                       n.warmup2 = rep(n.warmup2, n.chains),
                       n.thin = rep(n.thin, n.chains),
                       permutation = permutation.lst,
                       pars.oi = sampler$param_names_oi(),
                       dims.oi = sampler$param_dims_oi(),
                       fnames.oi = fnames.oi,
                       n.flatnames = n.flatnames) 
            fit <- new("stanfit",
                       model.name = object@model.name,
                       model.pars = m.pars, 
                       par.dims = p.dims, 
                       sim = sim,
                       # summary = summary,
                       # keep a record of the initial values 
                       inits = organize.inits(lapply(sim$samples, function(x) attr(x, "inits")), 
                                               m.pars, p.dims), 
                       stan.args = args.list,
                       .MISC = new.env()) 
             assign("stanmodel", object, envir = fit@.MISC)
             # keep a ref to avoid garbage collection
             # (see comments in fun stan.model)
             assign("date", date(), envir = fit@.MISC) 
             return(fit)
          }) 

