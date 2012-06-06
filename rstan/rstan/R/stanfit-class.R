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
         })


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

setGeneric(name = "extract",
           def = function(object, x) { standardGeneric("extract")}) 

setMethod("extract", "stanfit", 
          function(object, x) {
            cat("intend to return samples for parameters x.\n") 
          })  

#   if (!isGeneric('summary')) {
#     setGeneric(name = "summary",
#                def = function(object, ...) { 
#                        standardGeneric("summary")
#                      }) 
#   } 

setMethod("summary", signature = (object = "stanfit"), 
          function(object, probs, ...) { 

            if (missing(probs)) 
              probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  

            sampleshandle <- object@.fit$sampleshandle  
            vnames <- sampleshandle$param_names() 
            mnsd <- sampleshandle$get_mean_and_sd(vnames) 
            qs <- sampleshandle$get_quantiles(vnames, probs)  
            rhat <- sampleshandle$get_split_rhat(vnames) 
            ess <- sampleshandle$get_ess(vnames) 
         
            prob_in_percent <- paste(formatC(probs * 100,  
                                             digits = 1, 
                                             format = 'f', 
                                             drop0trailing = TRUE), 
                                     "%", sep = '')
            
            mqre <- cbind(do.call(rbind, mnsd), 
                          do.call(rbind, qs), 
                          do.call(rbind, rhat), 
                          do.call(rbind, ess)) 
            colnames(mqre) <- c("Mean", "SD", prob_in_percent, "Rhat", "ESS")
            mqre 
          })  


