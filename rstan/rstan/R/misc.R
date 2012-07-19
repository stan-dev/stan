
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
         });  
} 



data.preprocess <- function(data) { # , varnames) {
  # Preprocess the data (list or env) to list for stan
  # 
  # Args:
  #  data A list or environment: 
  #   * stop if there is NA; no-name lists; duplicate names  
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
    stop("data must be a list or environment")
  } 
 
  data <- lapply(data, 
                 FUN = function(x) {
 
                   ## Now we stop whenever we have NA in the data
                   ## since we do not know what variables are needed
                   ## at this point.
                   if (any(is.na(x))) {
                     stop("Stan does not support NA in the data.\n");
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
  lines <- readLines(con, n = -1L, warn = FALSE);
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
# model.code <- read.model.from.con('http://stan.googlecode.com/git/src/models/bugs_examples/vol1/dyes/dyes.stan');
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
      cat("init.v=", init.v, "\n") 
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

  if (!missing(sample.file)) {
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
  if (file.access(path, mode = 2) < 0)
    return(FALSE) 
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


stan.dump <- function(list, file, append = FALSE, 
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
    stop("stan.dump needs argument 'file', ",
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

  l2 <- NULL; 
  addnlpat <- paste0("(.{1,", width, "})(\\s|$)")
  for (v in list) {
    vv <- get(v, envir) 

    if (!is.numeric(vv))  next; 

    if (is.vector(vv)) {
      if (length(vv) == 1) {
        cat(v, " <- ", vv, "\n", file = file, sep = '')
        next;
      }
      str <- paste0(v, " <- \nc(", paste(vv, collapse = ', '), ")") 
      str <-  gsub(addnlpat, '\\1\n', str)
      cat(str, file = file) 
      l2 <- c(l2, v) 
      next; 
    }    

    if (is.matrix(vv) || is.array(vv)) { 
      l2 <- c(l2, v) 
      vvdim <- dim(vv)
      cat(v, " <- \n", file = file, sep = '')
      str <- paste0("structure(c(", paste(as.vector(vv), collapse = ', '), "),") 
      str <- gsub(addnlpat, '\\1\n', str)
      cat(str, 
          ".Dim = c(", paste(vvdim, collapse = ', '), "))\n", file = file, sep = '')
      next; 
    }
  }
  invisible(l2) 
} 

## test stan.dump simply
# a <- 1:3
# b <- 3
# c <- matrix(1:9, ncol = 3)
# d <- array(1:90, dim = c(9, 2, 5))
# stan.dump(c('a', 'b', 'c', 'd'), file = 'a.txt')

get.rhat.cols <- function(rhats) {
  # 
  # Args:
  #   rhat: a scale 
  #
  rhat.na.col <- get.rstan.options("plot.rhat.na.col")
  rhat.breaks <- get.rstan.options("plot.rhat.breaks")
  # print(rhat.breaks)
  rhat.colors <- get.rstan.options("plot.rhat.cols")
  
  sapply(rhats, 
         FUN = function(x) {
           if (is.na(x) || is.nan(x) || is.infinite(x))
             return(rhat.na.col)           
           for (i in 1:length(rhat.breaks)) {
             # cat("i=", i, "\n")
             if (x >= rhat.breaks[i]) 
               next;
             return(rhat.colors[i])
           }  
           get.rstan.options("plot.rhat.large.col")
		 })  
}


multi.print.plots <- function(ps, nrow = get.rstan.options("plot.nrow"), 
                                  ncol = get.rstan.options("plot.ncol")) {
  # plots a list of plots using grid.arrange 
  # 
  # Args:
  #  ps A list of plots obtained from ggplot or 
  #  those supported by grid.arrange 
  num.p <- length(ps)
  if (num.p < 1) return(NULL) 
  if (nrow == 1 && ncol == 1) {
    for (i in 1:num.p)
      print(ps[[i]])
  }
  stopifnot(require(gridExtra))
  start <- seq(1, num.p, by = nrow * ncol)
  end <- c(start[-1] - 1, num.p)
  for (i in 1:length(start)) {
    args <- c(ps[start[i]:end[i]], list(ncol = ncol, nrow = nrow))
    do.call(grid.arrange, args)
  }
  # virtualGrob
}


plot.pars0 <- function(mlu, cms, srhats, par.name, par.idx, 
                       plot = FALSE, prob = 0.8) {                
  # Plot a parameter (scale, vector, or array) with median, 
  # credible interval, and medians from separate chains, 
  # where par.name provides the parameter name and par.idx 
  # the indexes. par.idx could be empty for plotting a scale
  # parameter
  # 
  # Args:
  #   mlu: a list with elements of median, le, and ue, 
  #        computed from samples of all the chains for 
  #        all parameters.  For example, mlu$median
  #        is a vector of median for 5 parameters.
  #   cms: a list, each element of which is the medians of 
  #        separate chains for a parameter.
  #   srhats: a vector of split R hats for all parameters.
  #   par.name: parameter name, for example, beta.
  #   par.idx: parameter indexes, for example, [1], [2], [3].
  #   plot: TRUE -- render the plot; FALSE -- not. 
  #   prob: The probability of the interval, only used in 
  #         the title. So the caller should set prob 
  #         match what are in mlu
  # 
  # Returns: 
  #   A grob of ggplot

  num.par <- length(mlu[[1]])
  
  m.cols <- get.rstan.options("plot.chain.median.cols")
  srhat.cols = get.rhat.cols(srhats)
  # cat("srhat.cols=", srhat.cols, "\n")
  
  d <- data.frame(x = 1:num.par, 
                  y = mlu$median, 
                  le = mlu$le, 
                  ue = mlu$ue, 
                  cs = srhat.cols)
  # print(d)
  
  ## for later setting up all the colors manually
  cols.manual <- unique(c(srhat.cols, m.cols))
  names(cols.manual) <- cols.manual;

  lens <- sapply(cms, function(x) length(x))
  m.cols <- rep(m.cols, max(lens)) # in case m.cols's length is not enough
  colidx <- do.call(c, lapply(lens, function(n) 1:n)) 
  par.id <- do.call(c, lapply(1:length(lens), function(i) rep(i, lens[i])))
  d2 <- data.frame(x = par.id, 
                   y = do.call(c, cms),
                   col = m.cols[colidx])

  p1 <- ggplot() +
    geom_linerange(data = d, 
                   aes(x = x, ymin = le, ymax = ue, color = cs), 
                   size = 1, alpha = .8) +
    geom_point(data = d, 
               aes(x = x, y = y, colour = cs), 
               shape = 15, size = 3) + 
    geom_point(data = d2, 
               aes(x = x, y = y, colour = col), 
               shape = 4, size = 4) +
    scale_colour_manual(values = cols.manual) +
    ylab(par.name) + 
    opts(legend.position = "none", axis.title.x = theme_blank()) + 
    opts(title = paste0("Medians and ", probs2str(prob), " intervals")) +  
    scale_x_discrete(labels = par.idx)
  
  if (plot) print(p1)
  return(invisible(p1))
}

## test plot.pars0
# df <- data.frame(median = c(1,2,3), le = c(0.5, 1, 2), ue = c(2, 3, 4))
# cms <- list(c(1,2,3), c(4,5))
# p <- plot.pars0(df, cms, srhats = c(1.1,1.5,2), 
#                 par.name = "beta", 
#                 par.idx = c("[1]", "[2]", "[3]"))

read.rdump <- function(f) {
  # Read variables defined in an R dump file to an R list
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
        break; 
      } 
      res[i, j] <- 1
    } 
  } 
  res 
} 

flat.one.par <- function(n, d, col.major = FALSE) {
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
  #   pars:  the parameters of interess. This function assumes that
  #     pars are in names.   
  # Note: inside each parameter (vector or array), the sequence is in terms of
  #   col-major. That means if we have parameter alpha and beta, the dims
  #   of which are [2,2] and [2,3] respectively.  The whole parameter sequence
  #   are alpha[1,1], alpha[2,1], alpha[1,2], alpha[2,2], beta[1,1], beta[2,1],
  #   beta[1,2], beta[2,2], beta[1,3], beta[2,3]. 

  starts <- calc.starts(dims) 
  par.total.indexes <- function(par) {
    p <- match(par, fnames)
    if (!is.na(p)) {
      names(p) <- par 
      return(p) 
    } 
    p <- match(par, names) 
    idx <- starts[p] + seq(0, by = 1, length.out = num.pars(dims[[p]])) 
    names(idx) <- fnames[idx] 
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
#    return(list(mu = chain.id));
#  } 
#  b <- config.argss(3, c(100, 200), 10, 1, c("user", 1), fun1, seed = 3) 
#  print(b)
#  
#  


# colors friendly to color-blind 
# from http://colorbrewer2.org/
colp1 <- matrix(
  c(140, 81, 10, 216, 179, 101, 246, 232, 195, 
    245, 245, 245, 199, 234, 229, 90, 180, 172, 
    1, 102, 94),  
  byrow = TRUE, ncol = 3) 


colp2 <- matrix(
  c(178, 24, 43, 239, 138, 98, 253, 219, 199, 
    247, 247, 247, 209, 229, 240, 103, 169, 207, 
    33, 102, 172), 
  byrow = TRUE, ncol = 3) 

rstancolaa <- rgb(colp1, names = paste(1:7), maxColorValue = 255)
rstancolab <- rgb(colp1, alpha = 100, names = paste(1:7), maxColorValue = 255)
rstancolba <- rgb(colp2, names = paste(1:7), maxColorValue = 255)
rstancolbb <- rgb(colp2, alpha = 100, names = paste(1:7), maxColorValue = 255)


rstancolgrey <- rgb(matrix(c(247, 247, 247, 204, 204, 204, 150, 150, 150, 82, 82, 82),  
                           byrow = TRUE, ncol = 3), 
                    names = paste(1:4), maxColorValue = 255)

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

## some colors from package dichromat
rstancold <- 
  c("#990F0F", "#B22D2D", "#CC5252", "#E67E7E", "#FFB2B2", "#99700F", "#B28B2D",
    "#CCA852", "#E6C77E", "#FFE8B2", "#1F990F", "#3CB22D", "#60CC52", "#8AE67E",
    "#BCFFB2", "#710F99", "#8B2DB2", "#A852CC", "#C77EE6", "#E9B2FF", "#990F20",
    "#B22D3C", "#CC5260", "#E67E8A", "#FFB2BC") 

## summarize the chains merged and individually 
get.par.summary <- function(sim, n, probs = c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)) {
  n.warmup <- sim$n.warmup 
  ss <- lapply(1:sim$n.chains, function(i) sim$samples[[i]][[n]][-(1:n.warmup[i])]) 
  sumfun <- function(chain) c(mean(chain), sd(chain), quantile(chain, probs = probs))
  cs <- lapply(ss, sumfun)
  as <- sumfun(do.call(c, ss)) 
  list(summary = as, 
       c.summary = unlist(cs, use.names = FALSE), 
       ess = rstan:::rstan.ess(sim, n), 
       splitrhat = rstan:::rstan.splitrhat(sim, n)) 
} 

summary.sim <- function(sim, pars, probs = c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)) {
  probs.str <- probs2str(probs)
  pars <- if (missing(pars)) sim$pars.oi else check.pars(object, pars) 
  tidx <- rstan:::pars.total.indexes(sim$pars.oi, sim$dims.oi, sim$fnames.oi, pars) 
  tidx <- unlist(tidx, use.names = FALSE)
  s <- lapply(tidx, function(n) get.par.summary(sim, n, probs = probs))
  as <- lapply(s, function(x) c(x$summary, x$ess, x$splitrhat))  
  as <- do.call(rbind, as)
  rownames(as) <- sim$fnames.oi[tidx] 
  colnames(as) <- c("Mean", "SD", probs.str, "ESS", "Rhat") 
  cs <- lapply(s, function(x) x$c.summary) 
  cs <- do.call(rbind, cs) 
  dim(cs) <- c(length(tidx), 2 + length(probs), sim$n.chains) 
  dimnames(cs)[[1]] <- sim$fnames.oi[tidx] 
  dimnames(cs)[[2]] <- c("Mean", "SD", probs.str) 
  list(summary = as, c.summary = cs) 
}  

# a mimicking of bugs.plot.inferences in R2WinBUGS  
# 
# FIXME: to delete  ~/Desktop/bitb/yabbrep/winbugs
stan.plot.inferences <- function(sim, summary, pars, display.parallel = FALSE, ...) {
  # 
  # Args:
  #   sim: the sim list in stanfit object
  #   pars: parameters of interests
  #   display.parallel

  if (exists('windows'))  dev.fun <- windows 
  if (exists('X11'))  dev.fun <- X11 
  opt.dev <- options("device") 
  if (.Device %in% c("windows", "X11cairo")  ||
      (.Device=="null device" && identical(opt.dev, dev.fun))) {
    cex.names <- .7
    cex.axis <- .6
    cex.tiny <- .4
    cex.points <- .7
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
  mar.old <- par(mar = c(0, 0, 1, 0))

  plot(c(0, 1), c(-n.pars - .5, -.4), 
       ann = FALSE, bty = "n", xaxt = "n", yaxt = "n", type = "n")
  if (!is.R())
    options(warn = warn.settings)

  W <- max(strwidth(pars, cex = cex.names))
  # the max width of the variable names 

  # cex.names in defined at the beginning of this fun
  B <- (1 - W) / 3.8
  A <- 1 - 3.5 * B
  title <- if (display.parallel) "80% interval for each chain" else  "medians and 80% intervals"
  text (A, -.4, title, adj = 0, cex = cex.names)
  num.height <- strheight (1:9, cex = cex.tiny)

  for (k in 1:n.pars) { 
    k.dim <- sim$dims.oi[[pars[k]]] 
    k.aidx <- seq.array.ind(k.dim, col.major = TRUE) 
    
	# number of parameters we are going to plot for this 
	# particular vector/array parameter 

    index <- tidx[[k]] 
    k.num.p <- length(index) 
    J <- min(k.num.p, max.width)
    spacing <- 3.5 / max(J, standard.width)

    # the medians for all the kept samples merged 
    sprobs = c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
    mp <- match(0.5, sprobs) 
    i80p <- match(c(0.1, 0.9), sprobs) 
    med <- summary$summary[index, 2 + mp] 
    med <- array(med, dim = c(k.num.p, 1)) 
    i80 <- summary$summary[index, 2 + i80p] 
    i80 <- array(i80, dim = c(k.num.p, 2)) 
  
    med.chain <- summary$c.summary[index, 2 + mp, ]
    med.chain <- array(med.chain, dim = c(k.num.p, sim$n.chains)) 
    i80.chain <- summary$c.summary[index, 2 + i80p, ]
    i80.chain <- array(i80.chain, dim = c(k.num.p, 2, sim$n.chains))

    rng <- range(i80, i80.chain)
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
      text (A-B*.2, a+b*x, x, cex=cex.axis)
      lines (A+B*c(-.05,0), rep(a+b*x,2))
    }
    for (j in 1:J){
      if (display.parallel){
        for (m in 1:n.chains){
          interval <- a + b * i80.chain[j, , m]
          if (interval[2] - interval[1] < min.width)
            interval <- mean(interval) + c(-1,1)*min.width/2
          lines(A + B * spacing*rep(j+.6*(m-(n.chains+1)/2)/n.chains,2), interval, lwd=.5, col=m+1)
        }
      } else {
        lines (A + B * spacing * rep(j, 2), a + b * i80[j,], lwd = .5)
        for (m in 1:n.chains)
          points (A + B * spacing * j, a + b * med.chain[j, m], pch = 20, cex = cex.points, col = m + 1)
      } 

      # plot the dimension indexes for this parameter 
      if (length(k.dim) >= 1) { 
        # k.dim: the dimension of parameter k 
        for (m in 1:length(k.dim)) {
          index0 <- k.aidx[j, m] 
          if (j == 1)
            text(A+B*spacing*j, -k-height/2-.05-num.height*(m-1), index0, cex=cex.tiny)
          else if (index0 != k.aidx[j - 1, m] & (index0 %% (floor(log10(index0) + 1)) == 0))
            text(A+B*spacing*j, -k-height/2-.05-num.height*(m-1), index0, cex=cex.tiny)
        }
      }
    } 
    if (J < k.num.p) text (-.015, -k, "*", cex = cex.names, col = "red")
  } 
  par(mar = mar.old) 
} 


