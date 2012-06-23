
## Use an environment to keep some options, especially, 
## for plotting. 

.rstan.opt.env <- new.env() 


.init.rstan.opt.env <- function(e) {
  assign("plot.warmup.col", 19, e)
  assign("plot.kept.col", 20, e)
  assign("plot.chains.cols", 1:10, e)

  assign("plot.rhat.breaks", c(1.1, 1.2, 1.5, 2), e)
  assign("plot.rhat.cols", c("black", "black", "blue", "green", "red"), e)

  # when R-hat is NA, NaN, or Inf
  assign("plot.rhat.na.col", "red", e)
  # when R-hat is large than max(rhat.breaks) 
  assign("plot.rhat.large.col", "red", e)

  # for plot(stanfit)
  # the color for indicating the median of 
  # for samples from all the chains or 
  # separate chains
  assign("plot.chain.median.col", "black", e)

  # cat(".init.rstan.opt.env called.\n")
  return(e)
}

.init.rstan.opt.env(.rstan.opt.env)

get.rstan.options <- function(opt.name) {
  # Get an RStan option by name
  # Args:
  #   opt.name: the name of the options 
  # Return:
  #   A named list of option values. If the name 
  #   is not found, the value is NA. If opt.name 
  #   is just one name, the object is returned (not
  #   in a list). 

  e <- .rstan.opt.env
  if (length(as.list(e)) == 0)
    .init.rstan.opt.env(e)
  r <- mget(opt.name, envir = e, ifnotfound = NA)
  if (length(opt.name) == 1) return(r[[1]])
  r
}


rstan.options <- function(...) { 
  # e <- rstan:::.rstan.opt.env 
  e <- .rstan.opt.env 
  if (length(as.list(e)) == 0) 
    .init.rstan.opt.env(e) 

  a <-  list(...)
  len <- length(a) 
  if (len < 1) 
    return(e) 
  a.names <- names(a) 
  lapply(1:len, FUN = function(i) assign(a.names[i], a[[i]], e)) 
  e
} 

## test code 
# o <- rstan.options() 
# o <- rstan.options(b = 4)
# ls(o)
