
## Use an environment to keep some options, especially, 
## for plotting. 

.rstan.opt.env <- new.env() 


.init.rstan.opt.env <- function(e) {
  assign("plot.warmup.col", 19, e)
  # assign("plot.kept.col", 20, e)
  assign("plot.chains.cols", 1:10, e)

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

  # for plot(stanfit)
  # the color for indicating the median of 
  # for samples from all the chains or 
  # separate chains
  assign("plot.chain.median.cols", 
         c("green", "red", "yellow", "blue", "brown", "purple",
           "chocolate", "cyan", "coral"), e)

  # color for indicating or important info. 
  # for example, the color of star and text saying
  # the variable is truncated in stan.plot.inferences
  assign("rstan.alert.col", rgb(230, 85, 13, maxColorValue = 255), e)

  # color for plot chains in trace plot and stan.plot.inferences 
  assign("rstan.chain.cols", rstancolc, e)

  # cat(".init.rstan.opt.env called.\n")
  invisible(e)
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
  invisible(r)
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
  invisible(as.list(e))
} 

## test code 
# o <- rstan.options() 
# o <- rstan.options(b = 4)
# ls(o)
