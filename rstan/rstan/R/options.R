
## Use an evironment to keep some options, espectially, 
## for plotting. 

.rstan.opt.env <- new.env() 

.init.rstan.opt.env <- function(e) {
  assign("plot.warmup.color", 19, e) 
  assign("plot.kept.color", 20, e) 
  assign("plot.chains.colors", 1:10, e)
  
  # cat(".init.rstan.opt.env called.\n")
  return(e) 
} 


get.rstan.options <- function(opt.name) {
  # Get an Rstan option by name
  # Args:
  #   opt.name: the name of the options 
  # Return:
  #   A named list of option values. If the name 
  #   is not found, the value is NA. 

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

  a <-  as.list(match.call()) 

  len <- length(a) 
  if (len > 1) 
    sapply(2:len, FUN = function(i) assign(names(a)[[i]], a[[i]], e)) 
  e
} 
