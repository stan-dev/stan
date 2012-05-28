require(methods) 

setClass(Class = "stanmodel",
         representation = representation(
           model.name = "character", .modelmod = "list"
         ),  
         validity = function(object) {
           return(TRUE) 
         }); 


setMethod("show", "stanmodel",
          function(object) { 
            cat("Stan model: ", object@model.name, ".\n")  
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


setGeneric(name = "samples",
           def = function(object, data, n.chains = 1L, n.iter = 2000L, 
                          n.warmup = as.integer(n.iter / 2), 
                          thin = 1L, init.t = 'random', init.v = NULL, seed, 
                          sample.file, ...,
                          verbose = FALSE) { standardGeneric("samples")}) 

setMethod("samples", "stanmodel", 
          function(object, data, n.chains = 1L, n.iter = 2000L, 
                   n.warmup = as.integer(n.iter / 2), 
                   thin = 1L, init.t = "random", init.v = NULL, seed,
                   sample.file, ..., verbose = FALSE) {

            # check data and preprocess 
            data <- data.preprocess(data) 
 
            # assemble init and init_list 
            # init: how to set up the initial values (0, user, random)
            # init_list: user specified initial values list  
 
            init.t <- as.character(init.t)
            # cat("[in samples]: init.t=", init.t, "\n")
 
            if (!init.t %in% c("0", "user"))  
              init.t <- "random"; 
             
            args <- list(init = init.t, iter = n.iter, thin = thin) 
 
            if (init.t == 'user' && is.list(init.v)) 
              args$init_list <- init.v 
 
            if (!missing(seed)) 
              args$seed <- seed  
 
            if (!missing(n.warmup)) 
              args$warmup <- n.warmup 
 
            if (!missing(sample.file)) 
              args$sample_file <- sample.file  
                         
            # check inits and construct inits 
            # call the modules's nuts_command 
 
            # check if object is valid? (How?)

            nuts <- new(object@.modelmod$nuts, data, args) 
            invisible(nuts$call_nuts()) 
 
          })  

