require(methods)

setClass(Class = "stanmodel",
         representation = representation(
           model.name = "character",
           model.code = "character",
           .modelmod = "list"
         ), 
         validity = function(object) {
           return(TRUE)
         })


setMethod("show", "stanmodel",
          function(object) {
            cat("Stan model: ", object@model.name, ".\n" ,sep = '') 
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
          function(object, data = list(), pars = NA, n.chains = 1L, n.iter = 2000L,
                   n.warmup = floor(n.iter / 2),
                   n.thin = 1L, seed = sample.int(.Machine$integer.max, 1),
                   init.t = "random", init.v = NULL,
                   sample.file, verbose = FALSE, ...) {

            if (!is.sm.valid(object))
              stop("The compiled model in C++ is not valid any more")
            if (n.chains < 1) 
              stop("The number of chains (n.chains) is less than 1")

            # check data and preprocess
            if (!missing(data) && length(data) > 0)
              data <- data.preprocess(data)
            else
              data <- list()

            sampler <- new(object@.modelmod$sampler, data)
            if (!missing(pars) && !is.na(pars) && length(pars) > 0)
              sampler$update_param_oi(pars)

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
            summary <- summary.sim(sim)
            new("stanfit",
                model.name = object@model.name,
                model.pars = sampler$param_names(),
                model.dims = sampler$param_dims(),
                sim = sim,
                summary = summary,
                arg.lst = args.list,
                .MISC = list(stanmodel = object, date = date()))
                # keep a ref to avoid garbage collection
                # (see comments in fun stan.model)
          }) 

