
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

  ## seed: only one seed is needed by virtue of the RNG 

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
        stop("Initial value list mismatch number of chains") 
      if (!any(sapply(init.v, is.list))) {
        # print(init.v)
        stop("Initial value list is not a list of lists") 
      }
      init.vs <- init.v 
    } else { 
        stop("Wrong specification of initial values")
    } 
  } 

  argss <- vector("list", n.chains)  
  ## the name of arguments in the list need to 
  ## match those in include/rstan/stan_args.hpp 
  for (i in 1:n.chains)  
    argss[[i]] <- list(chain_id = i, 
                       iter = n.iters[i], thin = n.thins[i], 
                       warmup = n.warmups[i], init = init.ts[i]) 
                
    
  if (!missing(init.v) && !is.null(init.v))  
    for (i in 1:n.chains) 
      argss[[i]]$init_list = init.vs[[i]]   

  if (!missing(seed))  
      argss[[i]]$seed <- seed; 

  if (!missing(sample.file)) {
    if (n.chains == 1) 
        argss[[1]]$sample_file <- sample.file
    if (n.chains > 1) {
      for (i in 1:n.chains) 
        argss[[i]]$sample_file <- append.id(sample.file, i) 
    }
  }

  check.args(argss) 
  
  argss 
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
  # the indices. par.idx could be empty for plotting a scale
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
  #   par.idx: parameter indices, for example, [1], [2], [3].
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


## FIXME: a better way to check grid and ggplot2, 
## 
check.plot.pkgs <- function() {
   stopifnot(require("ggplot2"))
   stopifnot(require("grid"))    
   # stopifnot(require("gridExtra"))
} 

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
