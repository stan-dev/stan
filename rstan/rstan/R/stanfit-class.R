require(methods) 

setClass(Class = "stanfit",
         representation = representation(
           model.name = "character", 
           model.pars = "character", 
           num.chains = "numeric", 
           .fit = "list"
         ),  
         validity = function(object) {
           return(TRUE) 
         }); 


setMethod("show", "stanfit",
          function(object) { 
            cat("Stan fit: ", object@model.name, " with ", 
                object@num.chains, " chains.\n", sep = '')  
          })  

setMethod("plot", "stanfit",
          function(x, y, ...) { 
            cat("plot method of class stanfit.\n")  
          })  

setMethod("print", "stanfit",
          function(x, ...) { 
            cat("print method of class stanfit.\n")  
          })  

## add extract to the list of the methods that R knows.
#   setGeneric(name = "extract",
#              def = function(object, x) { standardGeneric("extract")}) 

#   setMethod("extract", "stanfit", 
#             function(object, x) {
#               cat("intend to return samples for parameters x.\n") 
#             })  

