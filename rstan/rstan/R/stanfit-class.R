# Part of the rstan package for an R interface to Stan 
# Copyright (C) 2012 Columbia University
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


setMethod("show", "stanfit", 
          function(object) {
            printstanfit(x = object, pars = object@sim$pars.oi)
          })  

printstanfit <- function(x, pars = x@sim$pars.oi, 
                         probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
                         digits.summary = 1, ...) { 
  s <- summary(x, pars, probs, ...)  
  cat("Inference for Stan model: ", x@model.name, '.\n', sep = '')
  cat(x@sim$n.chains, " chains: each with n.iter=", x@sim$n.iter, 
      "; n.warmup=", x@sim$n.warmup, "; n.thin=", x@sim$n.thin, "; ", 
      x@sim$n.save[1], " samples saved.\n\n", sep = '') 

  # round ESS to 0 decimal point 
  s$summary[, 'ESS'] <- round(s$summary[, 'ESS'], 0)

  print(round(s$summary, digits.summary), ...) 

  sampler <- attributes(x@sim$samples[[1]])$args$sampler 
  cat("\nSample were drawn using ", sampler, " at ", x@.MISC$date, ".\n", sep = '') 
  cat("For each parameter, ESS is a crude measure of effective samples size,\n") 
  cat("and Rhat is the potential scale reduction factor on split chains (at \n")
  cat("convergence, Rhat=1).\n")
}  
setMethod("print", "stanfit", printstanfit) 

if (!isGeneric("plot")) 
  setGeneric("plot", function(x, y, ...) standardGeneric("plot")) 

setMethod("plot", signature = "stanfit", 
          function(x, y, pars = y, display.parallel = FALSE, ...) {
            pars <- if (missing(pars) && missing(y)) x@sim$pars.oi else check.pars(x@sim, pars) 
            if (!exists("summary", envir = x@.MISC, inherits = FALSE))  
              assign("summary", summary.sim(x@sim), envir = x@.MISC)
            info <- list(model.name = x@model.name, model.date = x@.MISC$date) 
            stan.plot.inferences(x@sim, x@.MISC$summary, pars, info, display.parallel, ...) 
          }) 

setGeneric(name = "get.stancode",
           def = function(object, ...) { standardGeneric("get.stancode")}) 

setMethod('get.stancode', signature = "stanfit", 
          function(object, print = FALSE) {
            if (!exists("stanmodel", envir = object@.MISC, inherits = FALSE)) 
              stop("stanmodel is not found") 
            code <- object@.MISC$stanmodel@model.code
            if (print) cat(code, "\n") 
            invisible(code)
          }) 

setGeneric(name = 'get.stanmodel', 
           def = function(object, ...) { standardGeneric("get.stanmodel")})

setMethod("get.stanmodel", signature = "stanfit", 
          function(object) { 
            if (!exists("stanmodel", envir = object@.MISC, inherits = FALSE)) 
              stop("stanmodel is not found") 
            invisible(object@.MISC$stanmodel) 
          }) 

setGeneric(name = 'get.inits', 
           def = function(object, ...) { standardGeneric("get.inits")})

setMethod("get.inits", signature = "stanfit", 
          function(object) { invisible(object@inits) })

setGeneric(name = 'get.seed', 
           def = function(object, ...) { standardGeneric("get.seed")})

setMethod("get.seed", signature = "stanfit", 
          function(object) { invisible(object@stan.args[[1]]$seed) })

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
  if (length(pars_wo_ws) == 0) 
    stop("No parameter specified (pars is empty)")
  unique(pars_wo_ws) 
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
  ss <- mapply(gcks, sim$samples, sim$n.warmup2, sim$permutation,
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
  ss <- mapply(gcs, sim$samples, inc.warmup, sim$n.warmup2, 
               SIMPLIFY = FALSE, USE.NAMES = FALSE) 
  ss 
} 

par.traceplot <- function(sim, n, par.name, inc.warmup = TRUE) {
  # same n.thin, n.save, n.warmup2 for all the chains
  n.thin <- sim$n.thin[1] 
  n.warmup2 <- sim$n.warmup2[1] 
  n.save <- sim$n.save[1] 
  n.kept <- n.save - n.warmup2 
  yrange <- NULL 
  main <- paste("Trace of ", par.name) 
  chain.cols <- rstan.options("rstan.chain.cols")
  warmup.col <- rstan.options("rstan.warmup.bg.col") 
  if (inc.warmup) {
    id <- seq(1, by = n.thin, length.out = n.save) 
    for (i in 1:sim$n.chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]]) 
    }
    plot(c(1, id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations', ylab = "", main = main)
    rect(par("usr")[1], par("usr")[3], n.warmup2 * n.thin, par("usr")[4], 
         col = warmup.col, border = NA)
    for (i in 1:sim$n.chains) {
      lines(id, sim$samples[[i]][[n]], xlab = '', ylab = '', 
            lwd = 1, col = chain.cols[(i-1) %% 6 + 1]) 
    }
  } else {  
    idx <- n.warmup2 + 1:n.kept
    id <- seq((n.warmup2 + 1)* n.thin, by = n.thin, length.out = n.kept) 
    for (i in 1:sim$n.chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]][idx]) 
    }
    plot(c((n.warmup2 + 1), id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations (without warmup)', ylab = "", main = main)
    for (i in 1:sim$n.chains)  
      lines(id, sim$samples[[i]][[n]][idx], lwd = 1, 
            xlab = '', ylab = '', col = chain.cols[(i-1) %% 6 + 1]) 
  } 
} 

######

setGeneric(name = "extract",
           def = function(object, ...) { standardGeneric("extract")}) 

setMethod("extract", signature = "stanfit",
          definition = function(object, pars, permuted = FALSE, inc.warmup = TRUE) {
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

            n.kept <- object@sim$n.save - object@sim$n.warmup2
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
            if (!inc.warmup) n2 <- n2 - object@sim$n.warmup2[1] 
            dim(sssf) <- c(n2, object@sim$n.chains, length(tidx)) 
            dimnames(sssf) <- list(NULL, NULL, tidxnames) 
            sssf 
          })  

#   if (!isGeneric('summary')) {
#     setGeneric(name = "summary",
#                def = function(object, ...) { 
#                        standardGeneric("summary")
#                      }) 
#   } 

setMethod("summary", signature = "stanfit", 
          function(object, pars, 
                   probs = c(0.025, 0.25, 0.50, 0.75, 0.975), ...) { 
            # Summarize the samples (that is, compute the mean, SD, quantiles, for 
            # the samples in all chains and chains individually after removing
            # warmup samples, and ESS and split R hat for all the kept samples.)
            # 
            # Returns: 
            #   A list with elements:
            #   summary: the summary for all the kept samples 
            #   c.summary: the summary for individual chains. 
            # 
            # Note: 
            #   This function is a not straight in terms of implementation as it
            #   saves some standard summaries including ESS and Rhat in the environment
            #   of the object, which is used if it's available and created upon 
            #   the first time standard summary is called. . 
            #   In addition, the indexes complicate the implementation as internally
            #   we use column major indexes for vector/array parameters. But it might
            #   be better to use row major indexes for the output such as print.

            if (!exists("summary", envir = object@.MISC, inherits = FALSE)) 
              assign("summary", summary.sim(object@sim), envir = object@.MISC)
           
            pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
            if (missing(probs)) 
              probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  
            m <- match(probs, default.summary.probs())
            if (any(is.na(m))) {
              ss <-  summary.sim.quan(object@sim, pars, probs) 
              row.idx <- attr(ss, "row.major.idx") 
              col.idx <- attr(ss, "col.major.idx") 

              ss$ess <- object@.MISC$summary$ess[col.idx, drop = FALSE] 
              ss$rhat <- object@.MISC$summary$rhat[col.idx, drop = FALSE] 
              ss$mean <- object@.MISC$summary$msd[col.idx, 1, drop = FALSE] 
              ss$sd <- object@.MISC$summary$msd[col.idx, 2, drop = FALSE] 
              ss$sem <- object@.MISC$summary$sem[col.idx]  
              s1 <- cbind(ss$mean, ss$sem, ss$sd, 
                          ss$quan, ss$ess, ss$rhat)
              colnames(s1) <- c("Mean", "SE.Mean", "SD", colnames(ss$quan), 'ESS', 'Rhat')

              s2 <- combine.msd.quan(object@.MISC$summary$c.msd[col.idx, , , drop = FALSE], ss$c.quan) 

              idx2 <- match(row.idx, col.idx) 
              ss <- list(summary = s1[idx2, , drop = FALSE],
                         c.summary = s2[idx2, , , drop = FALSE]) 
              return(invisible(ss)) 
            }

            tidx <- pars.total.indexes(object@sim$pars.oi, 
                                       object@sim$dims.oi, 
                                       object@sim$fnames.oi, 
                                       pars) 
            tidx <- lapply(tidx, function(x) attr(x, "row.major.idx"))
            tidx <- unlist(tidx, use.names = FALSE)
            tidx.len <- length(tidx)

            ss <- object@.MISC$summary 

            s1 <- cbind(ss$msd[tidx, 1, drop = FALSE], 
                        ss$sem[tidx, drop = FALSE], 
                        ss$msd[tidx, 2, drop = FALSE], 
                        ss$quan[tidx, m, drop = FALSE], 
                        ss$ess[tidx, drop = FALSE],
                        ss$rhat[tidx, drop = FALSE])  
            pars.names <- rownames(ss$msd)[tidx] 
            qnames <- colnames(ss$quan)[m] 
            dim(s1) <- c(length(tidx), length(m) + 5) 
            rownames(s1) <- pars.names 
            colnames(s1) <- c("Mean", "SE.Mean", "SD", qnames, 'ESS', 'Rhat')
            s2 <- combine.msd.quan(ss$c.msd[tidx, , , drop = FALSE], ss$c.quan[tidx, m, , drop = FALSE]) 
            dim(s2) <- c(tidx.len, length(m) + 2, object@sim$n.chains)
            dimnames(s2) <- list(pars.names, c("Mean", "SD", qnames), NULL) 
            ss <- list(summary = s1, c.summary = s2)
            invisible(ss) 
          })  

if (!isGeneric("traceplot")) {
  setGeneric(name = "traceplot",
             def = function(object, ...) { standardGeneric("traceplot") }) 
} 

setMethod("traceplot", signature = "stanfit", 
          function(object, pars, inc.warmup = TRUE, ask = FALSE) { 

            pars <- if (missing(pars)) object@sim$pars.oi else check.pars(object@sim, pars) 
            tidx <- pars.total.indexes(object@sim$pars.oi, 
                                       object@sim$dims.oi, 
                                       object@sim$fnames.oi, 
                                       pars) 
            tidx <- lapply(tidx, function(x) attr(x, "row.major.idx"))
            tidx <- unlist(tidx, use.names = FALSE)
            par.mfrow.old <- par('mfrow')
            num.plots <- length(tidx) 
            if (num.plots %in% 2:4) par(mfrow = c(num.plots, 1)) 
            if (num.plots > 5) par(mfrow = c(4, 2)) 
            par.traceplot(object@sim, tidx[1], object@sim$fnames.oi[tidx[1]], 
                          inc.warmup = inc.warmup)
            if (num.plots > 8 && ask) ask.old <- devAskNewPage(ask = TRUE)
            if (num.plots > 1) { 
              for (n in 2:num.plots)
                par.traceplot(object@sim, tidx[n], object@sim$fnames.oi[tidx[n]], 
                              inc.warmup = inc.warmup)
            }
            if (ask) devAskNewPage(ask = ask.old)
            par(mfrow = par.mfrow.old)
            invisible(NULL) 
          })  

is.sf.valid <- function(sf) {
  # Similar to is.sm.valid  
  # This depends on currently that we return R_NilValue
  # in the `src` when calling cxxfunction. 
  return(rstan:::is.sm.valid(sf@.MISC$stanmodel)) 
} 
