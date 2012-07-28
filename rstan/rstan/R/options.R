
## Use an environment to keep some options, especially, 
## for plotting. 

.rstan.opt.env <- new.env() 


.init.rstan.opt.env <- function(e) {
  tmat <- matrix(c(254, 237, 222, 
                   253, 208, 162, 
                   253, 174, 107, 
                   253, 141, 60, 
                   230, 85, 13, 
                   166, 54, 3), 
                 byrow = TRUE, ncol = 3)

  rhat.cols <- rgb(tmat, alpha = 150, names = paste(1:nrow(tmat)),
                   maxColorValue = 255)

  assign("plot.rhat.breaks", c(1.1, 1.2, 1.5, 2), e)
  assign("plot.rhat.cols", rhat.cols, e)

  # when R-hat is NA, NaN, or Inf
  assign("plot.rhat.na.col", rhat.cols[6] , e)
  # when R-hat is large than max(rhat.breaks) 
  assign("plot.rhat.large.col", rhat.cols[6], e)

  # color for indicating or important info. 
  # for example, the color of star and text saying
  # the variable is truncated in stan.plot.inferences
  assign("rstan.alert.col", rgb(230, 85, 13, maxColorValue = 255), e)

  # color for plot chains in trace plot and stan.plot.inferences 
  assign("rstan.chain.cols", rstancolc, e)

  # color for shading the area of warmup trace plot
  assign("rstan.warmup.bg.col", rstan:::rstancolgrey[3], e)

  # boost lib path 
  rstan.inc.path  <- system.file('include', package = 'rstan')
  boost.lib.path <- file.path(rstan.inc.path, '/stanlib/boost_1.50.0') 
  assign("boost.lib", boost.lib.path, e) 

  # cat(".init.rstan.opt.env called.\n")
  invisible(e)
}

.init.rstan.opt.env(.rstan.opt.env)

rstan.options <- function(...) { 
  # Set/get options in RStan
  # Args: any options can be defined, using 'name = value' 
  #
  # e <- rstan:::.rstan.opt.env 
  e <- .rstan.opt.env 
  if (length(as.list(e)) == 0) 
    .init.rstan.opt.env(e) 

  a <-  list(...)
  len <- length(a) 
  if (len < 1) return(NULL) 
  a.names <- names(a) 
  # deal with the case that this function is called as 
  # rstan.options("a", "b")
  if (is.null(a.names)) {
    ns <- unlist(a) 
    if (!is.character(ns)) 
      stop("rstan.options only accepts arguments as `name=value'") 
    r <- mget(unlist(a), envir = e, ifnotfound = NA)
    if (length(r) == 1) return(r[[1]])
    return(invisible(r))
  } 
  # the case for, for example, 
  # rstan.options(a = 3, b = 4, "c")
  empty <- (a.names == '') 

  opt.names <- c(a.names[!empty], unlist(a[empty]))
  r <- mget(opt.names, envir = e, ifnotfound = NA)

  lapply(a.names[!empty], FUN = function(n) assign(n, a[[n]], e)) 

  if (length(r) == 1) return(r[[1]])
  invisible(r)
} 

## test code 
# o <- rstan.options() 
# o <- rstan.options(b = 4)
# ls(o)
