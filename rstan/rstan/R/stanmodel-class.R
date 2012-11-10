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

new_empty_stanfit <- function(stanmodel, model_pars = character(0), par_dims = list(), 
                              mode = 2L, sim = list(), 
                              inits = list(), stan_args = list()) { 
  new("stanfit",
      model_name = stanmodel@model_name,
      model_pars = model_pars, 
      par_dims = par_dims, 
      mode = mode,
      sim = sim, 
      inits = inits, 
      stan_args = stan_args, 
      stanmodel = stanmodel, 
      date = date(),
      .MISC = new.env()) 
} 

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
            stan_fit_cpp_module <- eval(call("$", mod, paste('stan_fit4', model_cppname, sep = ''))) 

            if (check_data) { 
              # allow data to be specified as a vector of character string 
              if (is.character(data)) {
                data <- try(mklist(data))
                if (is(data, "try-error")) {
                  message("failed to create the data; sampling not done") 
                  return(invisible(new_empty_stanfit(object)))
                }
              }
              # check data and preprocess
              if (!missing(data) && length(data) > 0) {
                data <- try(data_preprocess(data))
                if (is(data, "try-error")) {
                  message("failed to preprocess the data; sampling not done") 
                  return(invisible(new_empty_stanfit(object)))
                }
              } else data <- list()
            } 

            sampler <- try(new(stan_fit_cpp_module, data)) 
            if (is(sampler, "try-error")) {
              message('failed to create the sampler; sampling not done') 
              return(invisible(new_empty_stanfit(object)))
            } 
            on.exit({rm(sampler); invisible(gc())}) 

            m_pars = sampler$param_names() 
            p_dims = sampler$param_dims() 
            if (!missing(pars) && !is.na(pars) && length(pars) > 0) {
              sampler$update_param_oi(pars)
              m <- which(match(pars, m_pars, nomatch = 0) == 0)
              if (length(m) > 0) {
                message("no parameter ", paste(pars[m], collapse = ', '), "; sampling not done") 
                return(invisible(new_empty_stanfit(object, m_pars, p_dims, 2L))) 
              }
            }

            if (chains < 1) {
              message("the number of chains is less than 1; sampling not done") 
              return(invisible(new_empty_stanfit(object, m_pars, p_dims, 2L))) 
            }

            args_list <- try(config_argss(chains = chains, iter = iter,
                                          warmup = warmup, thin = thin,
                                          init = init, seed = seed, sample_file, ...))
   
            if (is(args_list, "try-error")) {
              message('error specification of arguments; sampling not done') 
              return(invisible(new_empty_stanfit(object, m_pars, p_dims, 2L))) 
            }

            n_save <- 1 + (iter - 1) %/% thin 
            # number of samples saved after thinning
            warmup2 <- 1 + (warmup - 1) %/% thin 
            n_kept <- n_save - warmup2 
            samples <- vector("list", chains)
            dots <- list(...)

            for (i in 1:chains) {
              # cat("[sampling:] i=", i, "\n")
              # print(args_list[[i]])
              if (is.null(dots$refresh) || dots$refresh > 0) 
                cat("SAMPLING FOR MODEL '", object@model_name, "' NOW (CHAIN ", i, ").\n", sep = '')
              samples[[i]] <- try(sampler$call_sampler(args_list[[i]])) 
              if (is(samples[[i]], "try-error") || is.null(samples[[i]])) {
                message("error occurred during calling the sampler; sampling not done") 
                return(invisible(new_empty_stanfit(object, m_pars, p_dims, 2L))) 
              }
            }

            inits_used = organize_inits(lapply(samples, function(x) attr(x, "inits")), 
                                        m_pars, p_dims) 

            # test_gradient mode: no sample 
            if (attr(samples[[1]], 'test_grad')) {
              sim = list(num_failed = samples)
              return(invisible(new_empty_stanfit(object, m_pars, p_dims, 1L, sim = sim, 
                                                 inits = inits_used, 
                                                 stan_args = args_list)))
            } 

            # perm_lst <- lapply(1:chains, function(id) rstan_seq_perm(n_kept, chains, seed, chain_id = id)) 
            ## sample_int is a little bit faster than our own rstan_seq_perm (one 
            ## reason is that the RNG used in R is faster),
            ## but without controlling the seed 
            perm_lst <- lapply(1:chains, function(id) sample.int(n_kept))

            fnames_oi <- sampler$param_fnames_oi()
            n_flatnames <- length(fnames_oi)
            sim = list(samples = samples,
                       iter = iter, thin = thin, 
                       warmup = warmup, 
                       chains = chains,
                       n_save = rep(n_save, chains),
                       warmup2 = rep(warmup2, chains),
                       thin = rep(thin, chains),
                       permutation = perm_lst,
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
                        # keep a record of the initial values 
                        inits = inits_used, 
                        stan_args = args_list,
                        stanmodel = object, 
                          # keep a ref to avoid garbage collection
                          # (see comments in fun stan_model)
                        date = date(),
                        .MISC = new.env()) 
             invisible(nfit)
          }) 

