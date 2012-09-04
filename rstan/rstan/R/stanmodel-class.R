setMethod("show", "stanmodel",
          function(object) {
            cat("S4 class stanmodel: `", object@model_name, "' coded as follows:\n" ,sep = '') 
            cat(object@model_code, "\n")
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

setGeneric(name = "get_cppcode", 
           def = function(object, ...) { standardGeneric("get_cppcode")})

setMethod("get_cppcode", "stanmodel", 
          function(object) {
            object@.modelmod$model_cppcode  
          }) 

setMethod("sampling", "stanmodel",
          function(object, data = list(), pars = NA, chains = 4, iter = 2000,
                   warmup = floor(iter / 2),
                   thin = 1, seed = sample.int(.Machine$integer.max, 1),
                   init = "random", check_data = TRUE, 
                   sample_file, verbose = FALSE, ...) {

            if (!is_sm_valid(object))
              stop("the compiled object from C++ code for this model is not valid")

            if (!is_dso_loaded(object@dso)) {
               grab_cxxfun(object@dso) 
               model_cppname <- object@.modelmod$model_cppname  
               mod <- Module(model_cppname, getDynLib(object@dso)) 
               stan_fit_cpp_module <- eval(call("$", mod, model_cppname))
               assign("sampler", stan_fit_cpp_module, envir = object@.modelmod)
            }

            if (chains < 1) 
              stop("the number of chains is less than 1")

            if (check_data) { 
              # allow data to be specified as a vector of character string 
              if (is.character(data)) data <- mklist(data) 

              # check data and preprocess
              if (!missing(data) && length(data) > 0) data <- data_preprocess(data)
              else data <- list()
            } 

            sampler <- new(object@.modelmod$sampler, data)
            m_pars = sampler$param_names() 
            p_dims = sampler$param_dims() 
            if (!missing(pars) && !is.na(pars) && length(pars) > 0) {
              sampler$update_param_oi(pars)
              m <- which(match(pars, m_pars, nomatch = 0) == 0)
              if (length(m) > 0) 
                stop("no parameter ", paste(pars[m], collapse = ', ')) 
            }

            args_list <- config_argss(chains = chains, iter = iter,
                                      warmup = warmup, thin = thin,
                                      init = init, seed = seed, sample_file, ...)
            n_save <- 1 + (iter - 1) %/% thin 
            # number of samples saved after thinning
            warmup2 <- 1 + (warmup - 1) %/% thin 
            n_kept <- n_save - warmup2 
            samples <- vector("list", chains)

            for (i in 1:chains) {
              # cat("[sampling:] i=", i, "\n")
              # print(args_list[[i]])
              cat("SAMPLING FOR MODEL '", object@model_name, "' NOW (CHAIN ", i, ").\n", sep = '')
              samples[[i]] <- sampler$call_sampler(args_list[[i]])
              if (is.null(samples[[i]])) 
                stop("error occurred during calling the sampler")
            }

            permutation.lst <-
              lapply(1:chains, function(id) sampler$permutation(c(n_kept, 1, id))) 

            fnames_oi <- sampler$param_fnames_oi()
            n_flatnames <- length(fnames_oi)
            sim = list(samples = samples,
                       iter = iter, thin = thin, 
                       warmup = warmup, 
                       chains = chains,
                       n_save = rep(n_save, chains),
                       warmup2 = rep(warmup2, chains),
                       thin = rep(thin, chains),
                       permutation = permutation.lst,
                       pars_oi = sampler$param_names_oi(),
                       dims_oi = sampler$param_dims_oi(),
                       fnames_oi = fnames_oi,
                       n_flatnames = n_flatnames) 
            fit <- new("stanfit",
                       model_name = object@model_name,
                       model_pars = m_pars, 
                       par_dims = p_dims, 
                       sim = sim,
                       # summary = summary,
                       # keep a record of the initial values 
                       inits = organize_inits(lapply(sim$samples, function(x) attr(x, "inits")), 
                                               m_pars, p_dims), 
                       stan_args = args_list,
                       .MISC = new.env()) 
             assign("stanmodel", object, envir = fit@.MISC)
             # keep a ref to avoid garbage collection
             # (see comments in fun stan_model)
             assign("date", date(), envir = fit@.MISC) 
             invisible(fit)
          }) 

