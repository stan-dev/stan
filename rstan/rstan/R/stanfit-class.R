setMethod("show", "stanfit", 
          function(object) {
            printstanfit(x = object, pars = object@sim$pars_oi)
          })  

printstanfit <- function(x, pars = x@sim$pars_oi, 
                         probs = c(0.025, 0.25, 0.5, 0.75, 0.975), 
                         digits_summary = 1, ...) { 
  s <- summary(x, pars, probs, ...)  
  cat("Inference for Stan model: ", x@model_name, '.\n', sep = '')
  cat(x@sim$chains, " chains: each with iter=", x@sim$iter, 
      "; warmup=", x@sim$warmup, "; thin=", x@sim$thin, "; ", 
      x@sim$n_save[1], " iterations saved.\n\n", sep = '') 

  # round n_eff to 0 decimal point 
  s$summary[, 'n_eff'] <- round(s$summary[, 'n_eff'], 0)

  print(round(s$summary, digits_summary), ...) 

  sampler <- attr(x@sim$samples[[1]], "args")$sampler 

  cat("\nSamples were drawn using ", sampler, " at ", x@date, ".\n", sep = '') 
  cat("For each parameter, n_eff is a crude measure of effective sample size,\n") 
  cat("and Rhat is the potential scale reduction factor on split chains (at \n")
  cat("convergence, Rhat=1).\n")
}  
setMethod("print", "stanfit", printstanfit) 

if (!isGeneric("plot")) 
  setGeneric("plot", function(x, y, ...) standardGeneric("plot")) 

setMethod("plot", signature(x = "stanfit", y = "missing"), 
          function(x, pars, display_parallel = FALSE) {
            pars <- if (missing(pars)) x@sim$pars_oi else check_pars(x@sim, pars) 
            if (!exists("summary", envir = x@.MISC, inherits = FALSE))  
              assign("summary", summary_sim(x@sim), envir = x@.MISC)
            info <- list(model_name = x@model_name, model_date = x@date) 
            stan_plot_inferences(x@sim, x@.MISC$summary, pars, info, display_parallel)
          }) 

setGeneric(name = "get_stancode",
           def = function(object, ...) { standardGeneric("get_stancode")}) 

setGeneric(name = "get_cppo_mode", 
           def = function(object, ...) { standardGeneric("get_cppo_mode") }) 

setMethod('get_cppo_mode', signature = "stanfit", 
           function(object) { 
             get_cppo(get_cxxflag(object@stanmodel)) 
           }) 

setMethod('get_stancode', signature = "stanfit", 
          function(object, print = FALSE) {
            code <- object@stanmodel@model_code
            if (print) cat(code, "\n") 
            invisible(code)
          }) 

setGeneric(name = 'get_stanmodel', 
           def = function(object, ...) { standardGeneric("get_stanmodel")})

setMethod("get_stanmodel", signature = "stanfit", 
          function(object) { 
            invisible(object@stanmodel) 
          }) 

setGeneric(name = 'get_inits', 
           def = function(object, ...) { standardGeneric("get_inits")})

setMethod("get_inits", signature = "stanfit", 
          function(object) { invisible(object@inits) })

setGeneric(name = 'get_seed', 
           def = function(object, ...) { standardGeneric("get_seed")})

setMethod("get_seed", signature = "stanfit", 
          function(object) { object@stan_args[[1]]$seed })

### HELPER FUNCTIONS
### 
check_pars <- function(sim, pars) {
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
  if (missing(pars)) return(sim$pars_oi) 
  pars_wo_ws <- gsub('\\s+', '', pars) 
  allpars <- c(sim$pars_oi, sim$fnames_oi) 
  m <- which(match(pars_wo_ws, allpars, nomatch = 0) == 0)
  if (length(m) > 0) 
    stop("no parameter ", paste(pars[m], collapse = ', ')) 
  if (length(pars_wo_ws) == 0) 
    stop("no parameter specified (pars is empty)")
  unique(pars_wo_ws) 
} 

get_kept_samples <- function(n, sim) {
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
  ss <- mapply(gcks, sim$samples, sim$warmup2, sim$permutation,
               SIMPLIFY = FALSE, USE.NAMES = FALSE) 
  do.call(c, ss) 
} 

get_samples <- function(n, sim, inc_warmup = TRUE) {
  # get chain samples
  # Returns:
  #   a list of chains for the nth parameter; each chain is an 
  #   element of the list.  
  gcs <- function(s, inc_warmup, nw) {
    if (inc_warmup)  return(s[[n]])
    else return(s[[n]][-(1:nw)]) 
  } 
  ss <- mapply(gcs, sim$samples, inc_warmup, sim$warmup2, 
               SIMPLIFY = FALSE, USE.NAMES = FALSE) 
  ss 
} 

par_traceplot <- function(sim, n, par.name, inc_warmup = TRUE) {
  # same thin, n_save, warmup2 for all the chains
  thin <- sim$thin[1] 
  warmup2 <- sim$warmup2[1] 
  n_save <- sim$n_save[1] 
  n_kept <- n_save - warmup2 
  yrange <- NULL 
  main <- paste("Trace of ", par.name) 
  chain_cols <- rstan_options("rstan_chain_cols")
  warmup_col <- rstan_options("rstan_warmup_bg_col") 
  if (inc_warmup) {
    id <- seq(1, by = thin, length.out = n_save) 
    for (i in 1:sim$chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]]) 
    }
    plot(c(1, id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations', ylab = "", main = main)
    rect(par("usr")[1], par("usr")[3], warmup2 * thin, par("usr")[4], 
         col = warmup_col, border = NA)
    for (i in 1:sim$chains) {
      lines(id, sim$samples[[i]][[n]], xlab = '', ylab = '', 
            lwd = 1, col = chain_cols[(i-1) %% 6 + 1]) 
    }
  } else {  
    idx <- warmup2 + 1:n_kept
    id <- seq((warmup2 + 1)* thin, by = thin, length.out = n_kept) 
    for (i in 1:sim$chains) {
      yrange <- range(yrange, sim$samples[[i]][[n]][idx]) 
    }
    plot(c((warmup2 + 1), id[length(id)]), yrange, type = 'n', 
         xlab = 'Iterations (without warmup)', ylab = "", main = main)
    for (i in 1:sim$chains)  
      lines(id, sim$samples[[i]][[n]][idx], lwd = 1, 
            xlab = '', ylab = '', col = chain_cols[(i-1) %% 6 + 1]) 
  } 
} 

######

setGeneric(name = 'get_adaptation_info', 
           def = function(object, ...) { standardGeneric("get_adaptation_info")})

setMethod("get_adaptation_info", 
          definition = function(object) {
            lai <- lapply(object@sim$samples, function(x) attr(x, "adaptation_info"))
            is_empty <- function(x) { 
              if (is.null(x)) return(TRUE) 
              if (is.character(x) && all(nchar(x) == 0)) return(TRUE)
              FALSE
            }
            if (all(sapply(lai, FUN = is_empty))) return(invisible(NULL))  
            invisible(lai) 
          }) 

setGeneric(name = "get_logposterior", 
           def = function(object, ...) { standardGeneric("get_logposterior")})


setMethod("get_logposterior", 
          definition = function(object, inc_warmup = TRUE) {
            llp <- lapply(object@sim$samples, function(x) x[['lp__']]) 
            if (inc_warmup) return(invisible(llp)) 
            invisible(mapply(function(x, w) x[-(1:w)], 
                             llp, object@sim$warmup2,
                             SIMPLIFY = FALSE, USE.NAMES = FALSE)) 
          }) 

setGeneric(name = 'get_sampler_params', 
           def = function(object, ...) { standardGeneric("get_sampler_params")}) 

setMethod("get_sampler_params", 
          definition = function(object, inc_warmup = TRUE) {
            ldf <- lapply(object@sim$samples, 
                          function(x) do.call(cbind, attr(x, "sampler_params")))   
            if (all(sapply(ldf, is.null))) return(invisible(NULL))  
            if (inc_warmup) return(invisible(ldf)) 
            invisible(mapply(function(x, w) x[-(1:w), , drop = FALSE], 
                             ldf, object@sim$warmup2, 
                             SIMPLIFY = FALSE, USE.NAMES = FALSE)) 
          }) 

setGeneric(name = "extract",
           def = function(object, ...) { standardGeneric("extract")}) 

setMethod("extract", signature = "stanfit",
          definition = function(object, pars, permuted = FALSE, inc_warmup = TRUE) {
            # Extract the samples in different forms for different parameters. 
            #
            # Args:
            #   object: the object of "stanfit" class 
            #   pars: the names of parameters (including other quantiles) 
            #   permuted: if TRUE, the returned samples are permuted without
            #     warming up. And all the chains are merged. 
            #   inc_warmup: if TRUE, warmup samples are kept; otherwise, 
            #     discarded. If permuted is TRUE, inc_warmup is ignored. 
            #
            # Returns:
            #   If permuted is TRUE, return an array (matrix) of samples with each
            #   column being the samples for a parameter. 
            #   If permuted is FALSE, return array with dimensions
            #   (# of iter (with or w.o. warmup), # of chains, # of flat parameters). 


            pars <- if (missing(pars)) object@sim$pars_oi else check_pars(object@sim, pars) 
            tidx <- pars_total_indexes(object@sim$pars_oi, 
                                       object@sim$dims_oi, 
                                       object@sim$fnames_oi, 
                                       pars) 

            n_kept <- object@sim$n_save - object@sim$warmup2
            fun1 <- function(par) {
              sss <- sapply(tidx[[par]], get_kept_samples, object@sim) 
              dim(sss) <- c(sum(n_kept), object@sim$dims_oi[[par]]) 
              sss 
            } 
           
            if (permuted) {
              slist <- lapply(pars, fun1) 
              names(slist) <- pars 
              return(slist) 
            } 

            tidx <- unlist(tidx, use.names = FALSE) 
            tidxnames <- object@sim$fnames_oi[tidx] 
            sss <- lapply(tidx, get_samples, object@sim, inc_warmup) 
            sss2 <- lapply(sss, function(x) do.call(c, x))  # concatenate samples from different chains
            sssf <- unlist(sss2, use.names = FALSE) 
  
            n2 <- object@sim$n_save[1]  ## assuming all the chains have equal iter 
            if (!inc_warmup) n2 <- n2 - object@sim$warmup2[1] 
            dim(sssf) <- c(n2, object@sim$chains, length(tidx)) 
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
            # warmup samples, and n_eff and split R hat for all the kept samples.)
            # 
            # Returns: 
            #   A list with elements:
            #   summary: the summary for all the kept samples 
            #   c_summary: the summary for individual chains. 
            # 
            # Note: 
            #   This function is not straight in terms of implementation as it
            #   saves some standard summaries including n_eff and Rhat in the
            #   environment of the object. The summaries and created upon 
            #   the first time standard summary is called and resued later if possible. 
            #   In addition, the indexes complicate the implementation as internally
            #   we use column major indexes for vector/array parameters. But it might
            #   be better to use row major indexes for the output such as print.

            if (!exists("summary", envir = object@.MISC, inherits = FALSE)) 
              assign("summary", summary_sim(object@sim), envir = object@.MISC)
           
            pars <- if (missing(pars)) object@sim$pars_oi else check_pars(object@sim, pars) 
            if (missing(probs)) 
              probs <- c(0.025, 0.25, 0.50, 0.75, 0.975)  
            m <- match(probs, default_summary_probs())
            if (any(is.na(m))) { # unordinary quantiles are requested 
              ss <-  summary_sim_quan(object@sim, pars, probs) 
              row_idx <- attr(ss, "row_major_idx") 
              col_idx <- attr(ss, "col_major_idx") 

              ss$ess <- object@.MISC$summary$ess[col_idx, drop = FALSE] 
              ss$rhat <- object@.MISC$summary$rhat[col_idx, drop = FALSE] 
              ss$mean <- object@.MISC$summary$msd[col_idx, 1, drop = FALSE] 
              ss$sd <- object@.MISC$summary$msd[col_idx, 2, drop = FALSE] 
              ss$sem <- object@.MISC$summary$sem[col_idx]  
              s1 <- cbind(ss$mean, ss$sem, ss$sd, 
                          ss$quan, ss$ess, ss$rhat)
              colnames(s1) <- c("mean", "se_mean", "sd", colnames(ss$quan), 'n_eff', 'Rhat')

              s2 <- combine_msd_quan(object@.MISC$summary$c_msd[col_idx, , , drop = FALSE], ss$c_quan) 

              idx2 <- match(row_idx, col_idx) 
              ss <- list(summary = s1[idx2, , drop = FALSE],
                         c_summary = s2[idx2, , , drop = FALSE])
              return(invisible(ss)) 
            }

            tidx <- pars_total_indexes(object@sim$pars_oi, 
                                       object@sim$dims_oi, 
                                       object@sim$fnames_oi, 
                                       pars) 
            tidx <- lapply(tidx, function(x) attr(x, "row_major_idx"))
            tidx <- unlist(tidx, use.names = FALSE)
            tidx_len <- length(tidx)

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
            colnames(s1) <- c("mean", "se_mean", "sd", qnames, 'n_eff', 'Rhat')
            s2 <- combine_msd_quan(ss$c_msd[tidx, , , drop = FALSE], ss$c_quan[tidx, m, , drop = FALSE]) 
            dim(s2) <- c(tidx_len, length(m) + 2, object@sim$chains)
            dimnames(s2) <- list(pars.names, c("mean", "sd", qnames), NULL) 
            ss <- list(summary = s1, c_summary = s2) 
            invisible(ss) 
          })  

if (!isGeneric("traceplot")) {
  setGeneric(name = "traceplot",
             def = function(object, ...) { standardGeneric("traceplot") }) 
} 

setMethod("traceplot", signature = "stanfit", 
          function(object, pars, inc_warmup = TRUE, ask = FALSE) { 

            pars <- if (missing(pars)) object@sim$pars_oi else check_pars(object@sim, pars) 
            tidx <- pars_total_indexes(object@sim$pars_oi, 
                                       object@sim$dims_oi, 
                                       object@sim$fnames_oi, 
                                       pars) 
            tidx <- lapply(tidx, function(x) attr(x, "row_major_idx"))
            tidx <- unlist(tidx, use.names = FALSE)
            par.mfrow.old <- par('mfrow')
            num.plots <- length(tidx) 
            if (num.plots %in% 2:4) par(mfrow = c(num.plots, 1)) 
            if (num.plots > 5) par(mfrow = c(4, 2)) 
            par_traceplot(object@sim, tidx[1], object@sim$fnames_oi[tidx[1]], 
                          inc_warmup = inc_warmup)
            if (num.plots > 8 && ask) ask.old <- devAskNewPage(ask = TRUE)
            if (num.plots > 1) { 
              for (n in 2:num.plots)
                par_traceplot(object@sim, tidx[n], object@sim$fnames_oi[tidx[n]], 
                              inc_warmup = inc_warmup)
            }
            if (ask) devAskNewPage(ask = ask.old)
            par(mfrow = par.mfrow.old)
            invisible(NULL) 
          })  

is_sf_valid <- function(sf) {
  # Similar to is_sm_valid  
  # This depends on currently that we return R_NilValue
  # in the `src` when calling cxxfunction. 
  return(rstan:::is_sm_valid(sf@stanmodel)) 
} 
