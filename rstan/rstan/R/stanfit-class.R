require(methods) 

setClass(Class = "stanfit",
         representation = representation(
           model.name = "character", 
           model.pars = "character", 
           model.dims = "list", 
           sim = "list", 
           summary = "list", 
           arg.lst = "list", 
           .MISC = "list"
         ),  
         validity = function(object) {
           return(TRUE) 
         })


setMethod("show", "stanfit",
          function(object) { 
            cat("Stan fit: ", object@model.name, " with ", 
                object@sim$n.chains, " chains.\n", sep = '')  
          })  

stanplot <- function(object, pars, prob = 0.8) { 
  pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
  probs = c(0.5, 0.5 + c(-prob, prob) * 0.5) 

  num.par <- length(pars)
  chains.v <- 1:object@sim$n.chains 

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
            invisible(stanplot(object = x, pars = pars, prob = prob, plot = TRUE))
          }) 

setMethod("print", signature = (x = "stanfit"),
          function(x, pars, 
                   probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
                   digits.summary = 3, ...) { 
            if (missing(pars)) pars <- x@model.pars
            s <- summary(x, pars, probs, ...)  
            print(round(s$summary, digits.summary), ...) 
          })  


### HELPER FUNCTIONS
### 
check.pars <- function(sim, pars) {
  #
  # Check if all parameter in pars is a valid parameter 
  # of the model 
  # 
  # Args:
  #   sim: The sim slot of class stanfit 
  #   pars:  a vector of character for parameter names
  # 
  # Returns:
  #   pars without white spaces, if any, if all are valid
  #   otherwise stop reporting error
  if (missing(pars)) return(sim$pars.oi) 
  pars_wo_ws <- gsub('\\s+', '', pars) 
  allpars <- c(sim$pars.oi, sim$fnames.oi) 
  m <- which(match(pars_wo_ws, allpars, nomatch = 0) == 0)
  if (length(m) > 0) 
    stop("No parameter ", paste(pars[m], collapse = ', ')) 
  pars_wo_ws
} 

get.kept.samples <- function(n, sim) {
  #
  # Args:
  #  sim: the sim slot in object stanfit 
  #  n: the nth parameter (starting from 1) 
  # Note: 
  #  samples from different chains are merged. 

  # get chain kept samples (gcks) 
  gcks <- function(s, nw, permutation) {
    s <- s[[n]][-(1:nw)] 
    s[permutation] 
  } 
  ss <- mapply(gcks, sim$samples, sim$n.warmup, sim$permutation,
               SIMPLIFY = FALSE, USE.NAMES = FALSE) 
  do.call(c, ss) 
} 

get.samples <- function(n, sim, inc.warmup = TRUE) {
  # get chain samples
  # Returns:
  #   a list of chains for the nth parameter; each chain is an 
  #   element of the list.  
  gcs <- function(s, inc.warmup, nw) {
    if (inc.warmup)  return(s[[n]])
    else return(s[[n]][-(1:nw)]) 
  } 
  ss <- mapply(gcs, sim$samples, inc.warmup, sim$n.warmup, 
               SIMPLIFY = FALSE, USE.NAMES = FALSE) 
  ss 
} 

par.traceplot <- function(sim, n, par.name, inc.warmup = TRUE) {
  # same n.thin, n.save, n.warmup for all the chains
  n.thin <- sim$n.thin[1] 
  n.warmup <- sim$n.warmup[1] 
  n.save <- sim$n.save[1] 
  n.kept <- n.save - n.warmup 
  yrange <- NULL 
  main <- paste("Trace plot of ", par.name) 
  
  if (inc.warmup) {
    id <- seq(1, by = n.thin, length.out = n.save) 
    for (i in 1:sim$n.chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]]) 
    }
    plot(c(1, id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations', ylab = "", main = main)
    rect(par("usr")[1], par("usr")[3], n.warmup * n.thin, par("usr")[4], 
         col = rstan:::rstancolgrey[3], border = NA)
    for (i in 1:sim$n.chains) {
      lines(id, sim$samples[[i]][[n]], xlab = '', ylab = '', 
            lwd = 1, col = rstancolc[(i-1) %% 6 + 1]) 
    }
  } else {  
    idx <- n.warmup + 1:n.kept
    id <- seq((n.warmup + 1)* n.thin, by = n.thin, length.out = n.kept) 
    for (i in 1:sim$n.chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]][idx]) 
    }
    plot(c((n.warmup + 1), id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations (without warmup)', ylab = par.name, main = main)
    for (i in 1:sim$n.chains)  
      lines(id, sim$samples[[i]][[n]][idx], lwd = 1, col = rstancolc[(i-1) %% 6 + 1]) 
  } 
} 

######

setGeneric(name = "extract",
           def = function(object, ...) { standardGeneric("extract")}) 

setMethod("extract", signature(object = "stanfit"),
          definition = function(object, pars, permuted = FALSE, inc.warmup = TRUE, ...) {
            # Extract the samples in different forms for different parameters. 
            #
            # Args:
            #   object: the object of "stanfit" class 
            #   pars: the names of parameters (including other quantiles) 
            #   permuted: if TRUE, the returned samples are permuted without
            #     warming up. And all the chains are merged. 
            #   inc.warmup: if TRUE, warmup samples are kept; otherwise, 
            #     discarded. If permuted is TRUE, inc.warmup is ignored. 
            #
            # Returns:
            #   If permuted is TRUE, return an array (matrix) of samples with each
            #   column being the samples for a parameter. 
            #   If permuted is FALSE, return array with dimensions
            #   (# of iter (with or w.o. warmup), # of chains, # of flat parameters). 


            pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
            tidx <- pars.total.indexes(object@sim$pars.oi, 
                                       object@sim$dims.oi, 
                                       object@sim$fnames.oi, 
                                       pars) 

            n.kept <- object@sim$n.save - object@sim$n.warmup 
            fun1 <- function(par) {
              sss <- sapply(tidx[[par]], get.kept.samples, object@sim) 
              dim(sss) <- c(sum(n.kept), object@sim$dims.oi[[par]]) 
              sss 
            } 
           
            if (permuted) {
              slist <- lapply(pars, fun1) 
              names(slist) <- pars 
              return(slist) 
            } 

            tidx <- unlist(tidx, use.names = FALSE) 
            tidxnames <- object@sim$fnames.oi[tidx] 
            sss <- lapply(tidx, get.samples, object@sim, inc.warmup) 
            sss2 <- lapply(sss, function(x) do.call(c, x))  # concatenate samples from different chains
            sssf <- unlist(sss2, use.names = FALSE) 
  
            n2 <- object@sim$n.save[1]  ## assuming all the chains have equal n.iter 
            if (!inc.warmup) n2 <- n2 - object@sim$n.warmup[1] 
            dim(sssf) <- c(n2, object@sim$n.chains, length(tidx)) 
            dimnames(sssf)[[3]] <- tidxnames
            sssf 
          })  

#   if (!isGeneric('summary')) {
#     setGeneric(name = "summary",
#                def = function(object, ...) { 
#                        standardGeneric("summary")
#                      }) 
#   } 

setMethod("summary", signature = (object = "stanfit"), 
          function(object, pars, 
                   probs = c(0.025, 0.25, 0.50, 0.75, 0.975), ...) { 
            pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
            if (missing(probs)) 
              probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  

            m <- match(probs, c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975))  
            if (any(is.na(m)))  return(summary.sim(object@sim, pars, probs)) 

            tidx <- pars.total.indexes(object@sim$pars.oi, 
                                       object@sim$dims.oi, 
                                       object@sim$fnames.oi, 
                                       pars) 
            tidx <- unlist(tidx, use.names = FALSE)
            stat.idx <- c(1:2, 2 + m) 
            stat.idx2 <- c(1:2, 2 + m, 12, 13) # including ess and rhat
           
            ss <- list(summary = object@summary$summary[tidx, stat.idx2], 
                       c.summary = object@summary$c.summary[tidx, stat.idx, ])
            invisible(ss) 
          })  

if (!isGeneric("traceplot")) {
  setGeneric(name = "traceplot",
             def = function(object, ...) { standardGeneric("traceplot") }) 
} 

setMethod("traceplot", signature = (object = "stanfit"), 
          function(object, pars, inc.warmup = TRUE, ask = FALSE) { 

            pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
            tidx <- pars.total.indexes(object@sim$pars.oi, 
                                       object@sim$dims.oi, 
                                       object@sim$fnames.oi, 
                                       pars) 
            tidx <- unlist(tidx, use.names = FALSE)
            par.mfrow.old <- par('mfrow')
            num.plots <- length(tidx) 
            if (num.plots %in% 2:4) par(mfrow = c(num.plots, 1)) 
            if (num.plots > 5) par(mfrow = c(4, 2)) 
            if (num.plots > 8) set.ask <- ask 
            par.traceplot(object@sim, tidx[1], object@sim$fnames.oi[1], 
                          inc.warmup = inc.warmup)
            if (ask) ask.old <- devAskNewPage(ask = ask)
            if (num.plots > 1) { 
              for (n in 2:num.plots)
                par.traceplot(object@sim, tidx[n], object@sim$fnames.oi[n], 
                              inc.warmup = inc.warmup)
            }
            if (ask) devAskNewPage(ask = ask.old)
            invisible(par(par.mfrow.old)) 
          })  

is.sf.valid <- function(sf) {
  # Similar to is.sm.valid  
  # This depends on currently that we return R_NilValue
  # in the `src` when calling cxxfunction. 
  return(rstan:::is.sm.valid(sf@.MISC$stanmodel)) 
} 
