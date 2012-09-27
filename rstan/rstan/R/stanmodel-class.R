setMethod("show", "stanmodel",
          function(object) {
            cat("S4 class stanmodel '", object@model_name, "' coded as follows:\n" ,sep = '') 
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
           def = function(object, ...) { standardGeneric("get_cppcode") })

setMethod("get_cppcode", "stanmodel", 
          function(object) {
            object@model_cpp$model_cppcode  
          }) 

setGeneric(name = "get_cxxflags", 
           def = function(object, ...) { standardGeneric("get_cxxflags") })
setMethod("get_cxxflags", "stanmodel", function(object) { object@dso@cxxflags }) 

setMethod("sampling", "stanmodel",
          function(object, data = list(), pars = NA, chains = 4, iter = 2000,
                   warmup = floor(iter / 2),
                   thin = 1, seed = sample.int(.Machine$integer.max, 1),
                   init = "random", check_data = TRUE, 
                   sample_file, verbose = FALSE, ...) {

            if (!is_sm_valid(object))
              stop(paste("the compiled object from C++ code for this model is invalid, possible reasons:\n",
                         "  - compiled with save_dso=FALSE;\n", 
                         "  - compiled on a different platform.", sep = '')) 

            model_cppname <- object@model_cpp$model_cppname 
            # cat("model_cppname=", model_cppname, '\n')
            if (!is_dso_loaded(object@dso)) {
              # load the dso if available 
              grab_cxxfun(object@dso) 
            } 
            # mod <- object@dso@.CXXDSOMISC$module 
            mod <- get("module", envir = object@dso@.CXXDSOMISC, inherits = FALSE) 
            stan_fit_cpp_module <- eval(call("$", mod, model_cppname)) 

            if (chains < 1) 
              stop("the number of chains is less than 1")

            if (check_data) { 
              # allow data to be specified as a vector of character string 
              if (is.character(data)) data <- mklist(data) 

              # check data and preprocess
              if (!missing(data) && length(data) > 0) data <- data_preprocess(data)
              else data <- list()
            } 

            sampler <- new(stan_fit_cpp_module, data) 
            on.exit({rm(sampler); invisible(gc())}) 

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

            # test_gradient mode: no sample 
            if (attr(samples[[1]], 'test_grad')) {
              sim = list(num_failed = samples)
              nfit <- new("stanfit",
                          model_name = object@model_name,
                          model_pars = m_pars, 
                          par_dims = p_dims, 
                          mode = 1L, 
                          sim = sim, 
                          inits = organize_inits(lapply(samples, function(x) attr(x, "inits")), 
                                                 m_pars, p_dims), 
                          stan_args = args_list,
                          stanmodel = object, 
                          # keep a ref to avoid garbage collection
                          # (see comments in fun stan_model)
                          date = date(),
                          .MISC = new.env()) 
              return(invisible(nfit)) 
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
            nfit <- new("stanfit",
                        model_name = object@model_name,
                        model_pars = m_pars, 
                        par_dims = p_dims, 
                        mode = 0L, 
                        sim = sim,
                        # summary = summary,
                        # keep a record of the initial values 
                        inits = organize_inits(lapply(sim$samples, function(x) attr(x, "inits")), 
                                               m_pars, p_dims), 
                        stan_args = args_list,
                        stanmodel = object, 
                          # keep a ref to avoid garbage collection
                          # (see comments in fun stan_model)
                        date = date(),
                        .MISC = new.env()) 
             # triger gc to really delete sampler, create from the sampler_mod.   
             # the issue here is that if sampler is removed later automatically by 
             # R's gabbage collector after the fx (the loaded dso) is removed, 
             # it will cause a segfault, which will crash R. 
             invisible(nfit)
          }) 

