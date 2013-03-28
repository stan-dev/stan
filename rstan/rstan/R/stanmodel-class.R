setMethod("show", "stanmodel",
          function(object) {
            cat("S4 class stanmodel '", object@model_name, "' coded as follows:\n" ,sep = '') 
            cat(object@model_code, "\n")
          }) 

setGeneric(name = 'optimizing',
           def = function(object, ...) { standardGeneric("optimizing")})

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

new_empty_stanfit <- function(stanmodel, miscenv = new.env(), 
                              model_pars = character(0), par_dims = list(), 
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
      .MISC = miscenv) 
} 

prep_call_sampler <- function(object) {
  if (!is_sm_valid(object))
    stop(paste("the compiled object from C++ code for this model is invalid, possible reasons:\n",
               "  - compiled with save_dso=FALSE;\n", 
               "  - compiled on a different platform;\n", 
               "  - not existed for reading csv files.", sep = '')) 
  if (!is_dso_loaded(object@dso)) {
    # load the dso if available 
    grab_cxxfun(object@dso) 
  } 
} 

setMethod("optimizing", "stanmodel", 
          function(object, data = list(), 
                   seed = sample.int(.Machine$integer.max, 1), 
                   init = 'random', check_data = TRUE, sample_file,
                   verbose = FALSE, ...) {
            prep_call_sampler(object)
            model_cppname <- object@model_cpp$model_cppname 
            mod <- get("module", envir = object@dso@.CXXDSOMISC, inherits = FALSE) 
            stan_fit_cpp_module <- eval(call("$", mod, paste('stan_fit4', model_cppname, sep = ''))) 
            if (check_data) {
              if (is.character(data)) {
                data <- try(mklist(data))
                if (is(data, "try-error")) {
                  message("failed to create the data; optimization not done") 
                  return(invisible(list(stanmodel = object)))
                }
              }
              if (!missing(data) && length(data) > 0) {
                data <- try(data_preprocess(data))
                if (is(data, "try-error")) {
                  message("failed to preprocess the data; optimization not done") 
                  return(invisible(list(stanmodel = object)))
                }
              } else data <- list()
            } 
            sampler <- try(new(stan_fit_cpp_module, data, object@dso@.CXXDSOMISC$cxxfun)) 
            if (is(sampler, "try-error")) {
              message('failed to create the optimizer; optimization not done') 
              return(invisible(list(stanmodel = object)))
            } 
            # on.exit({rm(sampler); invisible(gc())}) 
            m_pars <- sampler$param_names() 
            idx_of_lp <- which(m_pars == "lp__")
            m_pars <- m_pars[-idx_of_lp]
            p_dims <- sampler$param_dims()[-idx_of_lp]
            if (is.numeric(init)) init <- as.character(init)
            if (is.function(init)) init <- init()
            if (!is.list(init) && !is.character(init)) {
              message("wrong specification of initial values")
              return(invisible(list(stanmodel = object)))
            } 
            seed <- check_seed(seed, warn = 1)    
            if (is.null(seed))
              return(invisible(list(stanmodel = object)))
            args <- list(init = init, seed = seed, point_estimate = TRUE)
            if (!missing(sample_file) && is.na(sample_file)) 
              args$sample_file <- writable_sample_file(sample_file) 
            dotlist <- list(...)
            dotlist$test_grad <- FALSE # not to test gradient
            optim <- sampler$call_sampler(c(args, dotlist))
            names(optim$par) <- flatnames(m_pars, p_dims)
            optim["stanmodel"] <- object
            invisible(optim)
          }) 

setMethod("sampling", "stanmodel",
          function(object, data = list(), pars = NA, chains = 4, iter = 2000,
                   warmup = floor(iter / 2),
                   thin = 1, seed = sample.int(.Machine$integer.max, 1),
                   init = "random", check_data = TRUE, 
                   sample_file, verbose = FALSE, ...) {
            prep_call_sampler(object)
            model_cppname <- object@model_cpp$model_cppname 
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

            sampler <- try(new(stan_fit_cpp_module, data, object@dso@.CXXDSOMISC$cxxfun)) 
            sfmiscenv <- new.env()
            if (is(sampler, "try-error")) {
              message('failed to create the sampler; sampling not done') 
              return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv)))
            } 
            assign("stan_fit_instance", sampler, envir = sfmiscenv)
            # on.exit({rm(sampler); invisible(gc())}) 

            m_pars = sampler$param_names() 
            p_dims = sampler$param_dims() 
            if (!missing(pars) && !is.na(pars) && length(pars) > 0) {
              sampler$update_param_oi(pars)
              m <- which(match(pars, m_pars, nomatch = 0) == 0)
              if (length(m) > 0) {
                message("no parameter ", paste(pars[m], collapse = ', '), "; sampling not done") 
                return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv, m_pars, p_dims, 2L))) 
              }
            }

            if (chains < 1) {
              message("the number of chains is less than 1; sampling not done") 
              return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv, m_pars, p_dims, 2L))) 
            }

            args_list <- try(config_argss(chains = chains, iter = iter,
                                          warmup = warmup, thin = thin,
                                          init = init, seed = seed, sample_file, ...))
   
            if (is(args_list, "try-error")) {
              message('error specification of arguments; sampling not done') 
              return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv, m_pars, p_dims, 2L))) 
            }

            n_save <- 1 + (iter - 1) %/% thin 
            # number of samples saved after thinning
            warmup2 <- 1 + (warmup - 1) %/% thin 
            n_kept <- n_save - warmup2 
            samples <- vector("list", chains)
            dots <- list(...)
            mode <- if (!is.null(dots$test_grad) && dots$test_grad) "TESTING GRADIENT" else "SAMPLING"

            for (i in 1:chains) {
              # cat("[sampling:] i=", i, "\n")
              # print(args_list[[i]])
              if (is.null(dots$refresh) || dots$refresh > 0) 
                cat(mode, " FOR MODEL '", object@model_name, "' NOW (CHAIN ", i, ").\n", sep = '')
              samples_i <- try(sampler$call_sampler(args_list[[i]])) 
              if (is(samples_i, "try-error") || is.null(samples_i)) {
                message("error occurred during calling the sampler; sampling not done") 
                return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv,
                                                   m_pars, p_dims, 2L))) 
              }
              samples[[i]] <- samples_i
            }

            inits_used = organize_inits(lapply(samples, function(x) attr(x, "inits")), 
                                        m_pars, p_dims) 

            # test_gradient mode: no sample 
            if (attr(samples[[1]], 'test_grad')) {
              sim = list(num_failed = sapply(samples, function(x) x$num_failed))
              return(invisible(new_empty_stanfit(object, miscenv = sfmiscenv,
                                                 m_pars, p_dims, 1L, sim = sim, 
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
                       warmup2 = rep(warmup2, chains), # number of warmpu iters in n_save
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
                        .MISC = sfmiscenv) 
             invisible(nfit)
          }) 

