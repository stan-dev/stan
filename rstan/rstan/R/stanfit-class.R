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
  names(ps) <- pars 
  invisible(ps) 
} 

if (!isGeneric("plot")) 
  setGeneric("plot", function(x, y, ...) standardGeneric("plot")) 

setMethod("plot", signature = (x = "stanfit"), 
          function(x, y, pars = y, prob = 0.8, ...) {
            if (missing(y)) 
              y <- x@model.pars 
            check.plot.pkgs() 
            invisible(stanplot(object = x, pars = pars, prob = prob, plot = TRUE))
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

setGeneric(name = "extract",
           def = function(object, ...) { standardGeneric("extract")}) 

setMethod("extract", signature(object = "stanfit"), # , pars = "character"), 
          definition = function(object, pars, permuted = FALSE, inc.warmup = TRUE, ...) {
            # Obtain the samples of all chains from the C++ mcmc::chain object 
            #
            # Args:
            #   object: the object of "stanfit" class 
            #   pars: the names of parameters (including other quantities) 
            #   permuted: if TRUE, the returned samples are permuted without
            #     warming up. And all the chains are merged. 
            #   inc.warmup: if TRUE, warmup samples are kept; otherwise, 
            #     discarded.            
            #
            # Returns:
            #   If permuted is TRUE, return an array (matrix) of samples with each
            #   column being the samples for a parameter. 
            #   If permuted is FALSE, return a list, every element of which is
            #   samples of a chain and also a list. The list of chain's element
            #   is a vector of samples for one parameter. 
 
            sampleshandle <- object@.fit$sampleshandle  
            if (missing(pars)) {
              pars <- object@model.pars
            } else {
              pars <- check.pars(object, pars) 
            } 
            
            dimss <- sampleshandle$get_param_dimss(pars)
            if (permuted) {
              nk <- sampleshandle$num_kept_samples()
              s <- sampleshandle$get_kept_samples_permuted(pars, FALSE); # expand = F
              lapply(1:length(s),
                     FUN = function(i) {
                       if (length(dimss[[i]]) > 0)
                         dim(s[[i]]) <<- c(nk[i], dimss[[i]])
                     })
              return(s) 
            } 
  
            slist <- sampleshandle$get_samples(pars, inc.warmup, FALSE) # expand = F
            ns <- if(inc.warmup) { 
                    sapply(1:length(slist), 
                           FUN = function(i) sampleshandle$num_chain_samples(i)) 
                  } else {
                    sapply(1:length(slist), 
                           FUN = function(i) sampleshandle$num_chain_kept_samples(i)) 
                  } 
            # print(ns) 
            ## set the dim attributes for each parameter of each chain 
            lapply(1:length(slist), 
                   FUN = function(i) { # i index chain 
                     lapply(1:length(slist[[i]]), 
                            FUN = function(j) { # j index param 
                              if (length(dimss[[j]]) > 0)
                                dim(slist[[i]][[j]]) <<- c(ns[i], dimss[[j]])
                            }) 
                   }) 
            return(slist)
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

if (!isGeneric("traceplot")) {
  setGeneric(name = "traceplot",
             def = function(object, ...) { standardGeneric("traceplot") }) 
} 

setMethod("traceplot", signature = (object = "stanfit"), 
          function(object, pars, inc.warmup = TRUE, plot = TRUE) {
            check.plot.pkgs() 
            if (missing(pars)) {
              pars <- object@model.pars
            } else {
              pars <- check.pars(pars) 
            }

            sampleshandle <- object@.fit$sampleshandle  
            ss <- sampleshandle$get_samples(pars, inc.warmup, TRUE) # expand = T
            ss <- lapply(ss, function(x) do.call(cbind, x)) 
            pars <- colnames(ss[[1]]) 

            ## use the n.warmup and n.thin for the first chain 
            chain1_args <- sampleshandle$get_chain_stan_args(1) 
          
            # using names without '[' and ']', which seems to be 
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
           names(ps) <- pars 
           if (plot) multi.print.plots(ps) 
           return(invisible(ps)) 
          })  












