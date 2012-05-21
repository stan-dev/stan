require(methods) 

setClass(Class = "stanmodel",
         representation = representation(
           model.name = "character", .modelData = "list"
         ),  
         validity = function(object) {
           return(TRUE) 
         }); 


setMethod("show", "stanmodel",
          function(object) { 
            cat("show method of class stanmodel.\n")  
          })  

setMethod("plot", "stanmodel",
          function(x, y, ...) { 
            cat("plot method of class stanmodel.\n")  
          })  

setMethod("print", "stanmodel",
          function(x, ...) { 
            cat("print method of class stanmodel.\n")  
          })  

# add extract to the list of the methods that R knows.
setGeneric(name = "extract",
           def = function(object, x) { standardGeneric("extract")}) 

setMethod("extract", "stanmodel", 
          function(object, x) {
            cat("intend to return samples for parameters x.\n") 
          })  

# z <- new("stanmodel", .xData = list())
# print(z)
# extract(z) 


