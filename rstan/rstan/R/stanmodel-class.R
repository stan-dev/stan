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


setGeneric(name = "samples",
           def = function(object, data, n.chains = 1L, n.iter = 2000L, 
                          n.warmup = floor(n.iter / 2), 
                          n.thin = 1L, init.t = 'random', init.v = NULL, seed, 
                          sample.file, ...,
                          verbose = FALSE) { standardGeneric("samples")}) 

setMethod("samples", "stanmodel", 
          function(object, data, n.chains = 1L, n.iter = 2000L, 
                   n.warmup = floor(n.iter / 2), 
                   n.thin = 1L, init.t = "random", init.v = NULL, seed,
                   sample.file, ..., verbose = FALSE) {

            if (n.chains < 1)  
              stop("The number of chains (n.chains) must be postive") 

            # check data and preprocess 
            if (!missing(data)) 
              data <- data.preprocess(data) 
            else 
              data <- list()

            nuts <- new(object@.modelmod$nuts, data, n.chains)

            args.list <- config.argss(n.chains, n.iter, n.warmup, n.thin,
                                      init.t, init.v, seed, sample.file, ...)

            for (i in 1:n.chains) { 
              # print(args.list[[i]])
              nuts$call_sampler(args.list[[i]]) 
            }
  
  
            ## TODO, thinking about return nuts for this function. 
              
          })  

