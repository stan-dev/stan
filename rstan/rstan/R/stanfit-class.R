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

setMethod("print", signature = (x = "stanfit"),
          function(x, 
                   probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
                   pars, 
                   digits.summary = 3, 
                   ...) { 
            s <- summary(x, probs, pars, ...); 
            print(round(s, digits.summary), ...) 
          })  

setGeneric(name = "extract",
           def = function(object, pars) { standardGeneric("extract")}) 

setMethod("extract", "stanfit", 
          function(object, pars) {
            # Obtain the samples of all chains from the C++ mcmc::chain object 
            #
            # Args:
            #   object: the object of "stanfit" class 
            #   pars: the names of parameters (including other quantities) 
            #
            # Returns:
            #   A list, every element of which is samples of a chain and also
            #   a list. The list of chain's element is a vector of samples 
            #   for one parameter. 
 
            sampleshandle <- object@.fit$sampleshandle  
            if (missing(pars)) {
              pars <- object@model.pars
            } else {
              m <- which(match(pars, object@model.pars, nomatch = 0) == 0)
              if (length(m) > 0) 
                stop("error: no parameter ", paste(pars[m], collapse = ', ')) 
            } 
  
            lapply(sampleshandle$get_samples(pars),
                   FUN = function(x) do.call(cbind, x)) 

          })  

#   if (!isGeneric('summary')) {
#     setGeneric(name = "summary",
#                def = function(object, ...) { 
#                        standardGeneric("summary")
#                      }) 
#   } 


setMethod("summary", signature = (object = "stanfit"), 
          function(object, 
                   probs = c(0.025, 0.25, 0.50, 0.75, 0.975),  
                   pars, ...) {

            sampleshandle <- object@.fit$sampleshandle  
            if (missing(pars)) {
              pars <- object@model.pars
            } else {
              m <- which(match(pars, object@model.pars, nomatch = 0) == 0)
              if (length(m) > 0) 
                stop("error no parameter ", paste(pars[m], collapse = ', ')) 
            } 

            if (missing(probs)) 
              probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  
            
            # "%in%" <- function(x, table) match(x, table, nomatch = 0) > 0
            
            mnsd <- sampleshandle$get_mean_and_sd(pars) 
            qs <- sampleshandle$get_quantiles(pars, probs)  
            rhat <- sampleshandle$get_split_rhat(pars) 
            ess <- sampleshandle$get_ess(pars) 
            
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
  
