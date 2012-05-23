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
           def = function(object, data, n.chains = 1L, iter = 2000L, 
                          thin = 1L, init.t = 'random', init.v = NULL, seed, 
                          sample_file, ...,
                          verbose = FALSE) { standardGeneric("samples")}) 

setMethod("samples", "stanmodel", 
          function(object, data, n.chains = 1L, n.iter = 2000L, 
                   thin = 1L, init.t = 'random', init.v = NULL, seed,
                   sample_file, ..., verbose = FALSE) {
            # cat("[in samples]: init.t=", init.t, "\n")

            stan.samples(object, data, n.chains, n.iter, 
                         thin, init.t, init.v, seed, 
                         sample_file = sample_file, ...,
                         verbose) 
            # cat("call nuts to draw samples from the model. \n") 
          })  


# z <- new("stanmodel", .xData = list())
# print(z)
# extract(z) 


