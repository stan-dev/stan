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

##  refactor traceplot 
stanplot <- function(object, pars = object@model.pars, 
                     prob = 0.8, plot = FALSE) {
  
  if (missing(pars)) {
    pars <- object@model.pars
  } else {
    pars <- check.pars(object, pars) 
  } 
  sampleshandle <- object@.fit$sampleshandle  

  probs = c(0.5, 0.5 + c(-prob, prob) * 0.5) 

  num.par <- length(pars)
  chains.v <- 1:object@num.chains 

  ps <- vector("list", num.par) 
  for (i in 1:num.par) {
    par <- pars[i] 
    rhats <- sampleshandle$get_split_rhat(par) 
    mlu0 <- do.call(rbind, sampleshandle$get_quantiles(par, probs)) 
    cms <- lapply(chains.v, 
                  FUN = function(k) {
                    z <- sampleshandle$get_chain_quantiles(k, par, .5)
                    do.call(cbind, z) 
                  })
    cms <- data.frame(do.call(rbind, cms)) 
    # print(cms) 
    mlu <- list(median = mlu0[, 1], le = mlu0[, 2], ue = mlu0[, 3]) 
    par.idx <- gsub(par, "", names(rhats)) 
    rhats <- do.call(c, rhats) 
    # cat("rhats=", rhats, "\n") 
    ps[[i]] <- plot.pars0(mlu, cms, rhats, par, par.idx, prob = prob)
    
  } 
  if (plot) multi.print.plots(ps) 
  invisible(ps) 
} 

if (!isGeneric("plot")) 
  setGeneric("plot", function(x, y, ...) standardGeneric("plot")) 

setMethod("plot", signature = (x = "stanfit"), 
          function(x, y, prob = 0.8, ...) {
            if (missing(y)) 
              y <- x@model.pars 
            check.plot.pkgs() 
            invisible(stanplot(object = x, pars = y, prob = prob, plot = TRUE))
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

  if (missing(pars)) {
    pars <- object@model.pars
  } else {
    pars <- check.pars(object, pars) 
  } 
  sampleshandle <- object@.fit$sampleshandle  

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


traceplot <- function(object, pars, plot = TRUE) {

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

  n.warmup <- data.frame(x = chain1_args$n.warmup) 
  colw <- get.rstan.options("plot.warmup.col") 
  colk<- get.rstan.options("plot.kept.col") 

  tplot.l.cols <- get.rstan.options("plot.chains.cols") 


  num.par <- length(pars) 
  ps <- vector("list", num.par) 
  for (i in 1:num.par) {
    p <- ggplot()  
    if (!is.na(colw)) {
      p <- p + geom_rect(data = n.warmup, 
                         aes(xmin = -Inf, xmax = x, 
                             ymin = -Inf, ymax = Inf), fill = colw, alpha = 0.1)
    }
    if (!is.na(colk)) {
      p <- p + geom_rect(data = n.warmup, 
                         aes(xmax = Inf, xmin = x, 
                             ymin = -Inf, ymax = Inf, fill = colk), alpha = 0.1)
    }

    vname <- paste("par", i, sep = '')

    for (k in 1:object@num.chains) {
      p <- p + 
        geom_line(data = ss[[k]], 
                  aes_string(x = "Iterations", y = vname), 
                  color = tplot.l.cols[k]) + 
        opts(legend.position = "none") + 
        ylab(pars[i]) +
        opts(title = paste("Trace of ", pars[i])) 
        # FIXME: 
        # need points as well?  # geom_point() 
    }
    ps[[i]] <- p
  } 
  if (plot) multi.print.plots(ps) 
  return(invisible(ps)) 
} 


setMethod("traceplot", 
          signature(object = "stanfit", 
                    pars = "character"),
          traceplot) 
