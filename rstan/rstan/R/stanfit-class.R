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

## TODO: use get_chain_quantiles and get_quantiles
## in stanplot instead of summary 
##  refactor traceplot 
stanplot <- function(object, pars = object@model.pars) {
  
  probs = c(0.1, 0.5, 0.9)   
  cs <- chain.summary(object, chain.id = 1:object@num.chains, pars = pars, probs = probs)
  ss <- summary(object, pars = pars) 

  num.par <- length(pars)
  ps <- vector("list", num.par) 
  for (i in 1:num.par) {
    cs <- chain.summary(object, chain.id = 1:Object@num.chains, pars = pars[i], probs = probs)
    ss <- summary(object, pars = pars[i], probs = probs) 
    # mlu <- list(median = cs[["50%"]]
  } 
} 


setMethod("plot", "stanfit",
          function(x, y, ...) {
            if (missing(y)) 
              y <- object@model.pars 
            stanplot(object = x, pars = y, ...)
          }) 

setMethod("print", signature = (x = "stanfit"),
          function(x, 
                   probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
                   pars, 
                   digits.summary = 3, 
                   ...) { 
            if (missing(pars))  
              pars <- x@model.pars

            s <- summary(x, probs, pars, ...)  
            print(round(s, digits.summary), ...) 
          })  


### HELPER FUNCTIONS
### 
check.pars <- function(object, pars) {
  #
  # Check if all parameter in pars is a valid parameter 
  # of the model 
  # 
  # Args:
  #   object: S4 class stanfit 
  #   pars:  a vector of character for parameter names
  # 
  # Returns:
  #   pars without white spaces, if any, if all are valid
  #   otherwise stop reporting error
  pars_nows <- gsub('\\s+', '', pars) 
  allpars <- c(object@model.pars, object@.fit$sampleshandle$param_flat_names()) 
  m <- which(match(pars_nows, allpars, nomatch = 0) == 0)
  if (length(m) > 0) 
    stop("No parameter ", paste(pars[m], collapse = ', ')) 
  pars_nows
} 
  

get.all.chain.samples <- function(object, pars = object@model.pars) {
  
  # Get all the samples for all the chains
  # 
  # Args:
  #   object: S4 class stanfit 
  #   pars:  a vector of character for parameter names
  # 
  # Returns:
  #   a list, every element of which is the samples for 
  #   a chain 
  # 
  sampleshandle <- object@.fit$sampleshandle  
  ss <- vector("list", object@num.chains) 
  for (cid in 1:object@num.chains) {
    ss[[cid]] <- sampleshandle$get_chain_samples(cid, pars) 
  } 
  lapply(ss, FUN = function(x) do.call(cbind, x))
} 

### 

setGeneric(name = "extract",
           def = function(object, ...) { standardGeneric("extract")}) 

setMethod("extract", signature(object = "stanfit"), # , pars = "character"), 
          definition = function(object, pars, ...) {
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
              pars <- check.pars(object, pars) 
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

chain.summary <- function(object, chain.id = 1:object@num.chains, 
                          probs = c(0.025, 0.25, 0.50, 0.75, 0.975),  
                          pars, ...) {
  if (any(chain.id < 0) && any(chain.id > object@num.chains)) {
    stop("chain.id should be postive and less than the", 
         "number of chains.") 
  } 

  sampleshandle <- object@.fit$sampleshandle  
  if (missing(pars)) {
    pars <- object@model.pars
  } else {
    pars <- check.pars(object, pars) 
  } 

  if (missing(probs)) 
    probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  

  num.cid <- length(chain.id) 
  r <- vector("list", num.cid) 

  for (i in 1:num.cid) { 
    k <- chain.id[i] 
    mnsd <- sampleshandle$get_chain_mean_and_sd(k, pars) 
    qs <- sampleshandle$get_chain_quantiles(k, pars, probs)  
    r[[i]] <- cbind(do.call(rbind, mnsd), do.call(rbind, qs)) 
    colnames(r[[i]]) <- c("Mean", "SD", probs2str(probs)) 
  }
  names(r) <- paste("chain.", chain.id, sep = '')
  r 
} 

setMethod("chain.summary", 
          signature(object = "stanfit", 
                    chain.id = "numeric", 
                    probs = "numeric", 
                    pars = "character"), 
          chain.summary) 


setMethod("summary", signature = (object = "stanfit"), 
          function(object, 
                   probs = c(0.025, 0.25, 0.50, 0.75, 0.975),  
                   pars, ...) {

            sampleshandle <- object@.fit$sampleshandle  
            if (missing(pars)) {
              pars <- object@model.pars
            } else {
              pars <- check.pars(object, pars) 
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
            colnames(mqre) <- c("Mean", "SD", probs2str(probs), "Rhat", "ESS")
            invisible(mqre) 
          })  


traceplot <- function(object, pars) {

  if (missing(pars)) {
    pars <- object@model.pars
  } else {
    pars <- check.pars(pars) 
  }
  
  sampleshandle <- object@.fit$sampleshandle  
  ss <- get.all.chain.samples(object, pars) 

  ## use the n.warmup and n.thin for the first chain 
  chain1_args <- sampleshandle$get_chain_stan_args(1) 

  pars <- colnames(ss[[1]]) 

  # using tmpnames without '[' and ']', which seems to be 
  # difficult to deal with. 
  tmpnames <- c(paste("par", 1:length(pars), sep = ''), "Iterations")

  it.idx <- seq(from = 1, by = chain1_args$n.thin, length.out = nrow(ss[[1]]))
   
  ss <- lapply(ss, 
               FUN = function(x) {
                 x2 <- cbind(x, it.idx) 
                 colnames(x2) <- tmpnames 
                 data.frame(x2) 
               }) 

  tplot.b.col <- 
    do.call(c, get.rstan.options(c("plot.warmup.color", "plot.kept.color"))) 
  tplot.l.cols <- get.rstan.options("plot.chains.colors") 


  rects <- data.frame(xs = c(-Inf, chain1_args$n.warmup),  
                      xe = c(chain1_args$n.warmup, Inf), 
                      col = tplot.b.col) 


  for (i in 1:length(pars)) {
    p <- ggplot() + 
      geom_rect(data = rects, 
                aes(xmin = xs, xmax = xe, ymin = -Inf, ymax = Inf, fill = col), 
                alpha = 0.1)
    vname <- paste("par", i, sep = '')

    for (k in 1:object@num.chains) {
      p <- p + 
        geom_line(data = ss[[k]], 
                  aes_string(x = "Iterations", y = vname), 
                  color = tplot.l.cols[k]) + 
        opts(legend.position = "none") + 
        ylab(pars[i])   
        # FIXME: 
        # need points as well?  # geom_point() 
        # layout 
    }
    
    print(p)
  } 
} 


setMethod("traceplot", 
          signature(object = "stanfit", 
                    pars = "character"),
          traceplot) 
