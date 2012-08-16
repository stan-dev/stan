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



#   is.whole.number <- function(x) {
#     all.equal(x, round(x), check.attributes = FALSE) 
#   } 

#   as.integer.if.doable <- function(y) {
#     if (!is.numeric(y)) return(y) 
#     if (is.integer(y)) return(y) 
#     if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
#       storage.mode(y) <- "integer"  
#     return(y) 
#   } 
 
list.as.integer.if.doable <- function(x) {
  # change the storage mode from 'real' to 'integer' 
  # if applicable since by default R use real.
  #
  # Args:
  #  x: A list 
  # 
  # Note:
  # Ignore non-numeric vectors since we ignore
  # them in rlist_var_context 
  #
  lapply(x, 
         FUN = function(y) { 
           if (!is.numeric(y)) return(y) 
           if (is.integer(y)) return(y) 
           if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
             storage.mode(y) <- "integer"  
           return(y) 
         })
} 

mklist <- function(names, env = as.environment(-1)) {
  # Make a list using names 
  # Args: 
  #   names: character strings of names of objects 
  #   env: the environment to look for objects with names
  # Note: we use inherits = TRUE when calling mget 
  d <- mget(names, env, ifnotfound = NA, inherits = TRUE) 
  n <- which(is.na(d)) 
  if (length(n) > 0) {
    stop(paste("objects ", paste("'", names[n], "'", collapse = ', ', sep = ''), " not found", sep = ''))
  } 
  d 
} 


data.preprocess <- function(data) { # , varnames) {
  # Preprocess the data (list or env) to list for stan
  # 
  # Args:
  #  data: A list, an environment, or a vector of character strings for names
  #  of objects 
  #   * stop if there is NA; no-name lists; duplicate names  
  #   * stop if the objects given name is not found  
  #   * remove NULL, non-numeric elements 
  #   * change to integers when applicable 

  # 
  # if (is.environment(data)) {
    
  #   data <- mget(varnames, envir = data, mode = "numeric", 
  #                ifnotfound = list(NULL))
  #   data <- data[!sapply(data, is.null)]
  # }
  if (is.environment(data)) {
    data <- as.list(data) 
  } else if (is.list(data)) {
    v <- names(data)
    if (is.null(v)) 
      stop("data must be a named list")
          
    ## Stan would report error if variable is not found 
    ## from the list
    # if (any(nchar(v) == 0))  
    #   stop("unnamed variables in data list")
    # 
 
    if (any(duplicated(v))) {
      stop("Duplicated names in data list: ", 
           paste(v[duplicated(v)], collapse = " "))
    }
  } else {
    stop("data must be a list or an environment") 
  } 
 
  data <- lapply(data, 
                 FUN = function(x) {

                   ## change data.frame to array 
                   if (is.data.frame(x)) x <- data.matrix(x) 
 
                   ## Now we stop whenever we have NA in the data
                   ## since we do not know what variables are needed
                   ## at this point.
                   if (any(is.na(x))) {
                     stop("Stan does not support NA in the data.\n")
                   } 
 
                   # remove those not numeric data 
                   if (!is.numeric(x)) return(NULL) 
 
                   if (is.integer(x)) return(x) 
         
                   # change those integers stored as reals to integers 
                   if (isTRUE(all.equal(x, round(x), check.attributes = FALSE))) 
                     storage.mode(x) <- "integer"  
                   return(x) 
                 })   
 
  data[!sapply(data, is.null)] 
} 


read.model.from.con <- function(con) {
  lines <- readLines(con, n = -1L, warn = FALSE)
  paste(lines, collapse = '\n') 
} 

get.model.code <- function(file, model.code = '') {
  if (!missing(file)) {
    if (is.character(file)) {
      fname <- file
      file <- try(file(fname, "rt"))
      if (inherits(file, "try-error")) {
        stop(paste("Cannot open model file \"", fname, "\"", sep = ""))
      }
      on.exit(close(file))
    } else if (!inherits(file, "connection")) {
      stop("file must be a character string or connection")
    }
    model.code <- paste(readLines(file, warn = FALSE), collapse = '\n') 
  } else if (model.code == '') {  
    stop("Model file missing and empty model.code")
  } 
  model.code 
} 



# FIXEME: implement more check on the arguments 
check.args <- function(argss) {
  if (FALSE) stop() 
} 

#
# model.code <- read.model.from.con('http://stan.googlecode.com/git/src/models/bugs_examples/vol1/dyes/dyes.stan')
# cat(model.code)


append.id <- function(file, id, suffix = '.csv') {
  fname <- basename(file)
  fpath <- dirname(file)
  fname2 <- gsub("\\.csv[[:space:]]*$", 
                 paste("_", id, ".csv", sep = ''), 
                 fname)
  if (fname2 == fname) 
    fname2 <- paste(fname, "_", id, ".csv", sep = '')
  file.path(fpath, fname2)
}



config.argss <- function(n.chains, n.iter, n.warmup, n.thin, 
                        init.t, init.v,
                        seed, sample.file, ...) {

  n.iters <- rep(n.iter, n.chains)   
  n.thins <- rep(n.thin, n.chains)  
  n.warmups <- rep(n.warmup, n.chains) 

  init.t <- as.character(init.t)
  init.t[which(!init.t %in% c("0", "user"))] <- 'random'

  init.ts <- rep(init.t, n.chains)  
  init.vs <- vector("list", n.chains) 
  if (!missing(init.v) && !is.null(init.v)) {
    if (is.function(init.v)) {
      ## the function could take an argument named by chain.id 
      ## from 1 to num.chains. 
      if (any(names(formals(init.v)) == "chain.id")) {
        for (i in 1:n.chains)  
          init.vs[[i]] <- init.v(chain.id = i)
      } else {
        for (i in 1:n.chains)  
          init.vs[[i]] <- init.v() 
      } 
    } else if (is.list(init.v)) {
      if (length(init.v) != n.chains) 
        stop("Initial value list mismatchs number of chains") 
      if (!any(sapply(init.v, is.list))) {
        # print(init.v)
        stop("Initial value list is not a list of lists") 
      }
      init.vs <- init.v 
    } else { 
      # cat("init.v=", init.v, "\n") 
      stop("Wrong specification of initial values")
    } 
  } 

  ## only one seed is needed by virtue of the RNG 
  seed <- if (!missing(seed)) seed else sample.int(.Machine$integer.max, 1)

  argss <- vector("list", n.chains)  
  ## the name of arguments in the list need to 
  ## match those in include/rstan/stan_args.hpp 
  for (i in 1:n.chains)  
    argss[[i]] <- list(chain_id = i, 
                       iter = n.iters[i], thin = n.thins[i], seed = seed, 
                       warmup = n.warmups[i], init = init.ts[i]) 
    
  if (!missing(init.v) && !is.null(init.v))  
    for (i in 1:n.chains) 
      argss[[i]]$init_list = init.vs[[i]]   

  if (!missing(sample.file) && !is.na(sample.file)) {
    sample.file <- writable.sample.file(sample.file) 
    if (n.chains == 1) 
        argss[[1]]$sample_file <- sample.file
    if (n.chains > 1) {
      for (i in 1:n.chains) 
        argss[[i]]$sample_file <- append.id(sample.file, i) 
    }
  }
  dotlist <- list(...) 
  for (i in 1:n.chains) 
    argss[[i]] <- c(argss[[i]], dotlist) 
  check.args(argss) 
  argss 
} 

is.dir.writable <- function(path) {
  if (file.access(path, mode = 2) < 0) return(FALSE) 
  return(TRUE)
} 

writable.sample.file <- 
function(file, warning = TRUE, 
         wfun = function(x, x2) {
           paste('Warning: "', x, '" is not writable; use "', x2, '" instead.', sep = '')
         }) { 
  # Check if the path for file is writable, if not using tempdir() 
  # 
  # Args:
  #  file: The file interested. 
  #  warning: TRUE give a warning. 
  #  warningfun: A function that take two dirs for creating 
  #    the warning message. 
  # 
  # Returns:
  #  If the specified file is writable, return itself. 
  #  Otherwise, change the path to tempdir(). 
  
  dir <- dirname(file) 
  if (!is.dir.writable(dir)) { 
    dir2 <- tempdir()
    if (warning)
      cat(wfun(dir, dir2))
    return(file.path(dir2, basename(file)))
  } 
  file
} 


probs2str <- function(probs, digits = 1) {
  paste(formatC(probs * 100,  
                digits = digits, 
                format = 'f', 
                drop0trailing = TRUE), 
        "%", sep = '')
} 


stan.rdump <- function(list, file, append = FALSE, 
                       envir = parent.frame(),
                       width = options("width")$width) {
  # Dump an R list or environment for a model data 
  # to the R dump file that Stan supports.
  #
  # Args:
  #   list: a vector of character for all variables interested 
  #         (the same as in R's dump function) 
  #   file: the output file for dumping the variables. 
  #   append: then TRUE, the file is opened with 
  #           mode of appending; otherwise, a new file
  #           is created.  
  # 
  # Return:
 
  if (missing(file)) 
    stop("stan.rdump needs argument 'file', ",
         "into which the data are dumped.") 

  if (is.character(file)) {
    ex <- sapply(list, exists, envir = envir)
    if (!any(ex)) 
      return(invisible(character()))

    if (nzchar(file)) {
      file <- file(file, ifelse(append, "a", "w"))
      on.exit(close(file), add = TRUE)
    } else {
      file <- stdout()
    }
  }

  l2 <- NULL
  addnlpat <- paste0("(.{1,", width, "})(\\s|$)")
  for (v in list) {
    vv <- get(v, envir) 

    if (!is.numeric(vv))  next

    if (is.vector(vv)) {
      if (length(vv) == 1) {
        cat(v, " <- ", vv, "\n", file = file, sep = '')
        next
      }
      str <- paste0(v, " <- \nc(", paste(vv, collapse = ', '), ")") 
      str <-  gsub(addnlpat, '\\1\n', str)
      cat(str, file = file) 
      l2 <- c(l2, v) 
      next
    }    

    if (is.matrix(vv) || is.array(vv)) { 
      l2 <- c(l2, v) 
      vvdim <- dim(vv)
      cat(v, " <- \n", file = file, sep = '')
      str <- paste0("structure(c(", paste(as.vector(vv), collapse = ', '), "),") 
      str <- gsub(addnlpat, '\\1\n', str)
      cat(str, 
          ".Dim = c(", paste(vvdim, collapse = ', '), "))\n", file = file, sep = '')
      next
    }
  }
  invisible(l2) 
} 

## test stan.rdump simply
# a <- 1:3
# b <- 3
# c <- matrix(1:9, ncol = 3)
# d <- array(1:90, dim = c(9, 2, 5))
# stan.rdump(c('a', 'b', 'c', 'd'), file = 'a.txt')

get.rhat.cols <- function(rhats) {
  # 
  # Args:
  #   rhat: a scalar 
  #
  rhat.na.col <- rstan.options("plot.rhat.na.col")
  rhat.breaks <- rstan.options("plot.rhat.breaks")
  # print(rhat.breaks)
  rhat.colors <- rstan.options("plot.rhat.cols")

  sapply(rhats, 
         FUN = function(x) {
           if (is.na(x) || is.nan(x) || is.infinite(x))
             return(rhat.na.col)           
           for (i in 1:length(rhat.breaks)) {
             # cat("i=", i, "\n")
             if (x >= rhat.breaks[i]) next
             return(rhat.colors[i])
           }  
           rstan.options("plot.rhat.large.col")
		 })  
}

plot.rhat.legend <- function(x, y, height, p.cex) { 
  rhat.breaks <- rstan.options("plot.rhat.breaks")
  n.breaks <- length(rhat.breaks) 
  rhat.colors <- rstan.options("plot.rhat.cols")[1:n.breaks] 
  rhat.legend.txts <- c(paste("< ", rhat.breaks, sep = ''), 
                        paste(">= ", max(rhat.breaks), sep = ''))
  rhat.legend.cols <- c(rhat.colors, rstan.options('plot.rhat.large.col')) 
  rhat.legend.width <- strwidth(rhat.legend.txts) 
  rhat.rect.width <- strwidth("r-hat  ") 
  text(x, y, label = 'R-hat: ')  
  s1 <- strwidth('R-hat: ') 
  starts <- x + c(s1, s1 + cumsum(rhat.rect.width + rhat.legend.width)) 

  for (i in 1:length(rhat.legend.cols)) {
    rect(starts[i], y, starts[i] + rhat.rect.width, y + height, col = rhat.legend.cols[i], border = NA) 
    text(starts[i] + rhat.rect.width, y, label = rhat.legend.txts[i], cex = p.cex) 
  } 
} 
  

read.rdump <- function(f) {
  # Read data defined in an R dump file to an R list
  # 
  # Args:
  #   f: the file to be sourced
  # 
  # Returns:
  #   A list

  if (missing(f)) 
    stop("No file specified.")
  e <- new.env() 
  source(file = f, local = e)
  as.list(e)
} 


idx_col2rowm <- function(d) {
  # Suppose an iteration of samples for an array parameter is ordered by
  # col-major. This function generates the indexes that can be used to change
  # the sequences to row-major. 
  # Args:
  #   d: the dimension of the parameter 
  len <- length(d) 
  if (0 == len) return(1)  
  if (1 == len) return(1:d)  
  idx <- aperm(array(1:prod(d), dim = d)) 
  return(as.vector(idx)) 
} 


idx_row2colm <- function(d) {
  # What if it is row-major and we want col-major? 
  len <- length(d) 
  if (0 == len) return(1)  
  if (1 == len) return(1:d)  
  idx <- aperm(array(1:prod(d), dim = rev(d))) 
  return(as.vector(idx)) 
} 


seq.array.ind <- function(d, col.major = FALSE) {
  #
  # Generate an array of indexes for an array parameter 
  # in order of major or column. 
  #
  # Args:
  #   d: the dimensions of an array parameter, for example, 
  #     c(2, 3). 
  # 
  #   col.major: Determine what is the order of indexes. 
  #   If col.major = TRUE, for d = c(2, 3), return 
  #   [1, 1] 
  #   [2, 1] 
  #   [1, 2] 
  #   [2, 2] 
  #   [1, 3] 
  #   [2, 3] 
  #   If col.major = FALSE, for d = c(2, 3), return 
  #   [1, 1] 
  #   [1, 2] 
  #   [1, 3] 
  #   [2, 1] 
  #   [2, 2] 
  #   [2, 3] 
  # 
  # Returns: 
  #   If length of d is 0, return empty vector. 
  #   Otherwise, return an array of indexes, each
  #   row of which is an index. 
  # 
  # Note:
  #   R function arrayInd might be helpful sometimes. 
  # 
  if (length(d) == 0L)
    return(numeric(0L)) 
  total <- prod(d) 
  len <- length(d) 
  res <- array(1L, dim = c(total, len)) 
  jidx <- if (col.major) 1L:len else len:1L
  for (i in 2L:total) {
    res[i, ] <- res[i - 1, ]
    for (j in jidx) { 
      if (res[i - 1, j] < d[j]) {
        res[i, j] <- res[i - 1, j] + 1
        break
      } 
      res[i, j] <- 1
    } 
  } 
  res 
} 

flat.one.par <- function(n, d, col.major = FALSE) {
  # Return all the elemenetwise parameters for a vector/array
  # parameter. 
  # 
  # Args:
  #  n: Name of the parameter. For example, n = "alpha" 
  #  d: A vector indicates the dimensions of parameter n. 
  #     For example, d = c(2, 3).  d could be empty 
  #     as well when n is a scalar. 
  # 
  if (0 == length(d)) return(n)  
  nameidx <- seq.array.ind(d, col.major) 
  names <- apply(nameidx, 1, function(x) paste(n, "[", paste(x, collapse = ','), "]", sep = '')) 
  as.vector(names) 
} 


flatnames <- function(names, dims, col.major = FALSE) {
  if (length(names) == 1) 
    return(flat.one.par(names, dims[[1]], col.major = col.major))  
  nameslst <- mapply(flat.one.par, names, dims, 
                     MoreArgs = list(col.major = col.major), 
                     SIMPLIFY = FALSE,
                     USE.NAMES = FALSE) 
  if (is.vector(nameslst, "character")) 
    return(nameslst) 
  do.call(c, nameslst) 
} 

num.pars <- function(d) prod(d) 

calc.starts <- function(dims) {
  len <- length(dims) 
  s <- sapply(dims, function(d)  num.pars(d)) 
  cumsum(c(1, s))[1:len] 
} 


pars.total.indexes <- function(names, dims, fnames, pars) {
  # Obtain the total indexes for parameters (pars) in the 
  # whole sequences of names that is order by 'column major.' 
  # Args: 
  #   names: all the parameters names specifying the sequence of parameters 
  #   dims:  the dimensions for all parameters 
  #   fnames: all the parameter names specified by names and dims 
  #   pars:  the parameters of interest. This function assumes that
  #     pars are in names.   
  # Note: inside each parameter (vector or array), the sequence is in terms of
  #   col-major. That means if we have parameter alpha and beta, the dims
  #   of which are [2,2] and [2,3] respectively.  The whole parameter sequence
  #   are alpha[1,1], alpha[2,1], alpha[1,2], alpha[2,2], beta[1,1], beta[2,1],
  #   beta[1,2], beta[2,2], beta[1,3], beta[2,3]. In addition, for the col-majored
  #   sequence, an attribute named 'row.major.idx' is attached, which could
  #   be used when row major index is favored.

  starts <- calc.starts(dims) 
  par.total.indexes <- function(par) {
    p <- match(par, fnames)
    # note that here when `par' is a scalar, it would
    # match to one of `fnames'
    if (!is.na(p)) {
      names(p) <- par 
      attr(p, "row.major.idx") <- p 
      return(p) 
    } 
    p <- match(par, names) 
    idx <- starts[p] + seq(0, by = 1, length.out = num.pars(dims[[p]])) 
    names(idx) <- fnames[idx] 
    attr(idx, "row.major.idx") <- starts[p] + idx_col2rowm(dims[[p]]) - 1 
    idx
  } 
  idx <- lapply(pars, FUN = par.total.indexes) 
  names(idx) <- pars 
  idx 
} 

## simple test for pars.total.indexes 
#  names <- c('alpha', 'beta', 'gamma')
#  dims <- list(c(2,3), c(3,4,5), c(5))
#  fnames <- flatnames(names, dims, col.major = TRUE) 
#  pars.total.indexes(names, dims, fnames, c('gamma', 'alpha', 'beta')) 

#### temporary test code 
#  a <- config.argss(3, c(100, 200), 10, 1, "user", NULL, seed = 3) 
#  print(a) 
#  
#  fun1 <- function(chain.id) {
#    cat("chain.id=", chain.id)
#    return(list(mu = chain.id))
#  } 
#  b <- config.argss(3, c(100, 200), 10, 1, c("user", 1), fun1, seed = 3) 
#  print(b)
#  
#  
rstancolgrey <- rgb(matrix(c(247, 247, 247, 204, 204, 204, 150, 150, 150, 82, 82, 82),  
                           byrow = TRUE, ncol = 3), 
                    alpha = 100, 
                    names = paste(1:4), maxColorValue = 255)

# from http://colorbrewer2.org/, colorblind safe, 
# 6 different colors, diverging 
rstancolc <- rgb(matrix(c(230, 97, 1, 
                          153, 142, 195, 
                          84, 39, 136, 
                          241, 163, 64, 
                          216, 218, 235, 
                          254, 224, 182), 
                        byrow = TRUE, ncol = 3),
                 names = paste(1:6), maxColorValue = 255) 

default.summary.probs <- function() c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)

## summarize the chains merged and individually 
get.par.summary <- function(sim, n, probs = default.summary.probs()) {
  ss <- lapply(1:sim$n.chains, function(i) sim$samples[[i]][[n]][-(1:sim$n.warmup2[i])]) 
  msdfun <- function(chain) c(mean(chain), sd(chain))
  qfun <- function(chain) quantile(chain, probs = probs)
  c.msd <- unlist(lapply(ss, msdfun), use.names = FALSE) 
  c.quan <- unlist(lapply(ss, qfun), use.names = FALSE) 
  ass <- do.call(c, ss) 
  msd <- msdfun(ass) 
  quan <- qfun(ass) 
  list(msd = msdfun(ass), quan = qfun(ass), c.msd = c.msd, c.quan = c.quan) 
} 

# mean and sd 
get.par.summary.msd <- function(sim, n) { 
  ss <- lapply(1:sim$n.chains, function(i) sim$samples[[i]][[n]][-(1:sim$n.warmup2[i])]) 
  sumfun <- function(chain) c(mean(chain), sd(chain)) 
  cs <- lapply(ss, sumfun)
  as <- sumfun(do.call(c, ss)) 
  list(msd = as, c.msd = unlist(cs, use.names = FALSE)) 
} 

# quantiles 
get.par.summary.quantile <- function(sim, n, probs = default.summary.probs()) {
  ss <- lapply(1:sim$n.chains, function(i) sim$samples[[i]][[n]][-(1:sim$n.warmup2[i])]) 
  sumfun <- function(chain) quantile(chain, probs = probs)
  cs <- lapply(ss, sumfun)
  as <- sumfun(do.call(c, ss)) 
  list(quan = as, c.quan = unlist(cs, use.names = FALSE)) 
} 

combine.msd.quan <- function(msd, quan) {
  # Combine msd and quantiles for chain's summary 
  # Args:
  #   msd: the array for mean and sd with dim num.par * 2 * n.chains 
  #   cquan: the array for quantiles with dim num.par * n.quan * n.chains 
  dim1 <- dim(msd) 
  dim2 <- dim(quan) 
  if (any(dim1[c(1, 3)] != dim2[c(1, 3)])) 
    stop("numers of parameter/chains differ in msd and quan") 
  n.chains <- dim1[3] 
  n.par <- dim1[1] 
  n.stat <- dim1[2] + dim2[2] 
  par.names <- dimnames(msd)[[1]] 
  stat.names <- c(dimnames(msd)[[2]], dimnames(quan)[[2]]) 
  fun <- function(i) {
    # This is a bit ugly; one reason is that we need to 
    # deal with the case that dim1[1] = 1, in which 
    # a1 is a vector. 
    a1 <- msd[, , i] 
    a2 <- quan[, , i] 
    dim(a1) <- dim1[1:2] 
    dim(a2) <- dim2[1:2] 
    cbind(a1, a2)
  } 
  ll <- lapply(1:n.chains, fun) 
  twodnames <- dimnames(ll[[1]]) 
  msdquan <- array(unlist(ll), dim = c(n.par, n.stat, n.chains)) 
  dimnames(msdquan) <- list(par.names, stat.names, NULL) 
  msdquan 
} 

summary.sim <- function(sim, pars, probs = default.summary.probs()) {
  # cat("summary.sim is called.\n")
  probs.str <- probs2str(probs)
  probs.len <- length(probs) 
  pars <- if (missing(pars)) sim$pars.oi else check.pars(sim, pars) 
  tidx <- pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 
  tidx.rowm <- lapply(tidx, function(x) attr(x, "row.major.idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx.len <- length(tidx) 
  tidx.rowm <- unlist(tidx.rowm, use.names = FALSE)
  lmsdq <- lapply(tidx, function(n) get.par.summary(sim, n, probs)) 
  msd <- do.call(rbind, lapply(lmsdq, function(x) x$msd)) 
  quan <- do.call(rbind, lapply(lmsdq, function(x) x$quan)) 
  dim(msd) <- c(tidx.len, 2) 
  dim(quan) <- c(tidx.len, probs.len) 
  rownames(msd) <- sim$fnames.oi[tidx] 
  rownames(quan) <- sim$fnames.oi[tidx] 
  colnames(msd) <- c("Mean", "SD") 
  colnames(quan) <- probs.str 

  c.msd <- do.call(rbind, lapply(lmsdq, function(x) x$c.msd)) 
  c.quan <- do.call(rbind, lapply(lmsdq, function(x) x$c.quan)) 
  dim(c.msd) <- c(tidx.len, 2, sim$n.chains) 
  dim(c.quan) <- c(tidx.len, probs.len, sim$n.chains) 

  dimnames(c.msd) <- list(sim$fnames.oi[tidx], c("Mean", "SD"), NULL) 
  dimnames(c.quan) <- list(sim$fnames.oi[tidx], probs.str, NULL)

  ess <-  array(sapply(tidx, function(n) rstan.ess(sim, n)), dim = c(tidx.len, 1)) 
  rhat <- array(sapply(tidx, function(n) rstan.splitrhat(sim, n)), dim = c(tidx.len, 1)) 

  ss <- list(msd = msd, sem = msd[, 2] / sqrt(ess), 
             c.msd = c.msd, quan = quan, c.quan = c.quan, 
             ess = ess, rhat = rhat) 
  attr(ss, "row.major.idx") <- tidx.rowm 
  attr(ss, "col.major.idx") <- tidx
  ss
}  

summary.sim.quan <- function(sim, pars, probs = default.summary.probs()) {
  # cat("summary.sim is called.\n")
  probs.str <- probs2str(probs)
  probs.len <- length(probs) 
  pars <- if (missing(pars)) sim$pars.oi else check.pars(sim, pars) 
  tidx <- pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 
  tidx.rowm <- lapply(tidx, function(x) attr(x, "row.major.idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx.len <- length(tidx) 
  tidx.rowm <- unlist(tidx.rowm, use.names = FALSE)
  lquan <- lapply(tidx, function(n) get.par.summary.quantile(sim, n, probs)) 
  quan <- do.call(rbind, lapply(lquan, function(x) x$quan)) 
  dim(quan) <- c(tidx.len, probs.len) 
  rownames(quan) <- sim$fnames.oi[tidx] 
  colnames(quan) <- probs.str 

  c.quan <- do.call(rbind, lapply(lquan, function(x) x$c.quan)) 
  dim(c.quan) <- c(tidx.len, probs.len, sim$n.chains) 
  dimnames(c.quan) <- list(sim$fnames.oi[tidx], probs.str, NULL)

  ss <- list(quan = quan, c.quan = c.quan)
  attr(ss, "row.major.idx") <- tidx.rowm 
  attr(ss, "col.major.idx") <- tidx
  ss
}  

summary.sim.ess <- function(sim, pars) {
  pars <- if (missing(pars)) sim$pars.oi else check.pars(sim, pars) 
  tidx <- pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 
  tidx.rowm <- lapply(tidx, function(x) attr(x, "row.major.idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx.rowm <- unlist(tidx.rowm, use.names = FALSE)
  ess <- sapply(tidx, function(n) rstan.ess(sim, n)) 
  names(ess) <- sim$fnames.oi[tidx]
  attr(ess, "row.major.idx") <- tidx.rowm
  attr(ess, "col.major.idx") <- tidx
  ess
} 

summary.sim.rhat <- function(sim, pars) {
  pars <- if (missing(pars)) sim$pars.oi else check.pars(sim, pars) 
  tidx <- pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 
  tidx.rowm <- lapply(tidx, function(x) attr(x, "row.major.idx"))
  tidx <- unlist(tidx, use.names = FALSE)
  tidx.rowm <- unlist(tidx.rowm, use.names = FALSE)
  rhat <- sapply(tidx, function(n) rstan.splitrhat(sim, n)) 
  names(rhat) <- sim$fnames.oi[tidx]
  attr(rhat, "row.major.idx") <- tidx.rowm
  attr(rhat, "col.major.idx") <- tidx
  rhat 
} 


organize.inits <- function(inits, pars, dims) {
  # obtain a list of inital values for each chain in sim
  # Args: 
  #   inits: a list of vectors, each vector is the 
  #     inits for a chain 
  n.chains <- length(inits) 
  starts <- calc.starts(dims) 
  tmpfun <- function(x) {
    lst <- lapply(1:length(pars),  
                  function(i) { 
                    len <- num.pars(dims[[i]]) 
                    if (1 == len) return(x[starts[i]]) 
                    y <- x[starts[i] + (1:len) - 1] 
                    dim(y) <- dims[[i]] 
                    return(y) 
                  })
    names(lst) <- pars 
    lst 
  } 
  lapply(inits, tmpfun) 
} 

# ported from bugs.plot.inferences in R2WinBUGS  
# 
stan.plot.inferences <- function(sim, summary, pars, model.info, display.parallel = FALSE, ...) {
  # 
  # Args:
  #   sim: the sim list in stanfit object
  #   pars: parameters of interest
  #   model.info: names list with elements model.name and model.date 
  #   display.parallel

  alert.col <- rstan.options("rstan.alert.col")
  chain.cols <- rstan.options("rstan.chain.cols")
  chain.cols.len <- length(chain.cols) 

  if (exists('windows'))  dev.fun <- windows 
  if (exists('X11'))  dev.fun <- X11 
  opt.dev <- options("device") 
  if (.Device %in% c("windows", "X11cairo")  ||
      (.Device=="null device" && identical(opt.dev, dev.fun))) {
    cex.names <- .7
    cex.axis <- .6
    cex.tiny <- .4
    cex.points <- .7
    # the standard number of parameters in an array parameters. 
    # we have this so that even the # of parameters are less than
    # 30, we still have equal space between parameters. 
    standard.width <- 30
    max.width <- 40
    min.width <- .02
  } else {
    cex.names <- .7
    cex.axis <- .6
    cex.tiny <- .4
    cex.points <- .3
    standard.width <- 30
    max.width <- 40
    min.width <- .01
  }
  pars <- if (missing(pars)) sim$pars.oi else check.pars(sim, pars) 
  n.pars <- length(pars) 
  n.chains <- sim$n.chains
 
  tidx <- pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 

  ## if in Splus, suppress printing of warnings during the plotting.
  ## otherwise a warning is generated 
  if (!is.R()) {
    warn.settings <- options("warn")[[1]]
    options (warn = -1)
  }
  height <- .6
  # mar: c(bottom, left, top, right)
  mar.old <- par(mar = c(1, 0, 1, 0))

  plot(c(0, 1), c(-n.pars - .5, -.4), 
       ann = FALSE, bty = "n", xaxt = "n", yaxt = "n", type = "n")
  if (!is.R())
    options(warn = warn.settings)

  # plot the model general information 
  header <- paste("Stan model '", model.info$model.name, "' (", n.chains, 
                  " chains: n.iter=", sim$n.iter, "; n.burnin=", sim$n.warmup, 
                  "; n.thin=", sim$n.thin, ") fitted at ",
                  model.info$model.date, sep = '') 
  # side: (1=bottom, 2=left, 3=top, 4=right)
  mtext(header, side = 3, outer = TRUE, line = -1, cex = .7)

  W <- max(strwidth(pars, cex = cex.names))
  # the max width of the variable names 

  # cex.names is defined at the beginning of this fun
  B <- (1 - W) / 3.8
  A <- 1 - 3.5 * B
  title <- if (display.parallel) "80% interval for each chain" else  "medians and 80% intervals"
  text(A, -.4, title, adj = 0, cex = cex.names)
  num.height <- strheight (1:9, cex = cex.tiny) * 1.2

  truncated <- FALSE 
  for (k in 1:n.pars) { 
    text (0, -k, pars[k], adj = 0, cex = cex.names)

    k.dim <- sim$dims.oi[[pars[k]]] 
    k.dim.len <- length(k.dim)
    k.aidx <- seq.array.ind(k.dim, col.major = FALSE) 
    
    # the index for the parameters in the whole 
    # sequences of parameters 
    index <- attr(tidx[[k]], "row.major.idx")  

	# number of parameters we could plot for this 
	# particular vector/array parameter 
    k.num.p <- length(index) 

    # number of parameter we would plot
    J <- min(k.num.p, max.width)
    spacing <- 3.5 / max(J, standard.width)

    # the medians for all the kept samples merged 
    sprobs = default.summary.probs()  
    mp <- match(0.5, sprobs) 
    i80p <- match(c(0.1, 0.9), sprobs) 
    med <- summary$quan[index, mp] 
    med <- array(med, dim = c(k.num.p, 1)) 
    i80 <- summary$quan[index, i80p] 
    i80 <- array(i80, dim = c(k.num.p, 2)) 
    rhats <- summary$rhat 
    rhats.cols <- get.rhat.cols(rhats) 
  
    med.chain <- summary$c.quan[index, mp, ]
    med.chain <- array(med.chain, dim = c(k.num.p, sim$n.chains)) 
    i80.chain <- summary$c.quan[index, i80p, ]
    i80.chain <- array(i80.chain, dim = c(k.num.p, 2, sim$n.chains))

    rng <- if (display.parallel) range(i80, i80.chain) else range(i80)
    p.rng <- pretty(rng, n = 2)
    b <- height / (max(p.rng) - min(p.rng))
    a <- -(k + height / 2) - b * p.rng[1]
    lines(A + c(0, 0), -k + 0.5 * height * c(-1, 1)) 
    
    # plot a line at zero (if zero is in the range of the mini-plot)
    if (min(p.rng) < 0 & max(p.rng) > 0) {
      lines(A + B * spacing * c(0, J + 1), 
            rep(a, 2), lwd = .5, col = "gray")
    }
	# plot the breaks of the axis
    for (x in p.rng){
      text(A - B * .2, a + b * x, x, cex = cex.axis)
      lines(A + B * c(-.05, 0), rep(a + b * x, 2))
    }
    for (j in 1:J){
      if (display.parallel){
        for (m in 1:n.chains){
          interval <- a + b * i80.chain[j, , m]

          # When the interval is too tiny, we use the min.width instead
          # of the real one. 
          if (interval[2] - interval[1] < min.width)
            interval <- mean(interval) + c(-.5, .5) * min.width
          segments(x0 = A + B * spacing * (j + .6 *(m - (n.chains + 1) / 2) / n.chains), 
                   y0 = interval[1], y1 = interval[2], lwd = .5, 
                   col = chain.cols[(m-1) %% chain.cols.len + 1]) 
        }
      } else {
        lines(A + B * spacing * rep(j, 2), a + b * i80[j,], lwd = .5)
        for (m in 1:n.chains)
          points(A + B * spacing * j, a + b * med.chain[j, m], 
                 pch = 20, cex = cex.points, 
                 col = chain.cols[(m-1) %% chain.cols.len + 1])
      } 

      # draw an indicator for Rhat
      # (xleft, ybottom, xright, ytop)
       
      if (k.dim.len == 0) 
        rect(A + B * spacing * (j - .5), -k - height / 2 - 0.05 + num.height * .5, 
             A + B * spacing * (j + .5), -k - height / 2 - 0.05 - num.height * .5, col = rhats.cols[j], border = NA) 

      # plot the dimension indexes for this parameter 
      if (k.dim.len  >= 1) { 
        rect(A + B * spacing * (j - .5), -k - height / 2 - 0.05 + num.height * .5, 
             A + B * spacing * (j + .5), -k - height / 2 - 0.05 - num.height * (k.dim.len - .5), col = rhats.cols[j], border = NA) 

        # k.dim: the dimension of parameter k 
        for (m in 1:k.dim.len) {
          index0 <- k.aidx[j, m] 
          if (j == 1)
            text(A+B*spacing*j, -k-height/2-.05-num.height*(m-1), index0, cex=cex.tiny)
          else if (index0 != k.aidx[j - 1, m] & (index0 %% (floor(log10(index0) + 1)) == 0))
            text(A+B*spacing*j, -k-height/2-.05-num.height*(m-1), index0, cex=cex.tiny)

          # Note for `(index0 %% (floor(log10(index0) + 1)) == 0) in the above condition.
          # When 10 <= index0 <= 99, floor(log10(index0) + 1) == 2,
          # so that one index would be drawn out of two consecutive. 
          # That is, we would have 10, 12, 14, 16, etc. 
          # Similarly, when 100 <= index0 <= 999, we draw one out of three
          # though in the case, we do not draw them at all since the max is  
          # 40.  
        }
      }
    } 
    if (J < k.num.p) {
      text (-.015, -k, "*", cex = cex.names, col = alert.col)
      truncated <- TRUE
    } 
  } 
  invisible(par(mar = mar.old)) 
  if (truncated) {
    text(0, -n.pars - .5, "*  array truncated for lack of space", adj = 0, cex = cex.names, col = alert.col)
  } 
} 

legitimate.model.name <- function(name) {
  # To make model name be a valid name in C++. 
  return("anon_model")
} 

boost.url <- function() {"http://www.boost.org/users/download/"} 

makeconf.path <- function() {
  arch <- .Platform$r_arch
  if (arch == '') 
    return(file.path(R.home(component = 'etc'), 'Makeconf'))
  return(file.path(R.home(component = 'etc'), arch, 'Makeconf'))
} 
