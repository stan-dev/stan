## Use an environment to keep some options, especially, 
## for plotting. 

.rstan_opt_env <- new.env() 

init_rstan_opt_env <- function(e) {
  tmat <- matrix(c(254, 237, 222, 
                   253, 208, 162, 
                   253, 174, 107, 
                   253, 141, 60, 
                   230, 85, 13, 
                   166, 54, 3), 
                 byrow = TRUE, ncol = 3)

  rhat_cols <- rgb(tmat, alpha = 150, names = paste(1:nrow(tmat)),
                   maxColorValue = 255)

  assign("plot_rhat_breaks", c(1.1, 1.2, 1.5, 2), e)
  # in this default setting, 
  # if rhat < rhat.breaks[i], the color is rhat_cols[i]
  assign("plot_rhat_cols", rhat_cols, e)

  # when R-hat is NA, NaN, or Inf
  assign("plot_rhat_nan_col", rhat_cols[6] , e)
  # when R-hat is large than max(rhat.breaks) 
  assign("plot_rhat_large_col", rhat_cols[6], e)

  # color for indicating or important info. 
  # for example, the color of star and text saying
  # the variable is truncated in stan_plot_inferences
  assign("rstan_alert_col", rgb(230, 85, 13, maxColorValue = 255), e)

  # color for plot chains in trace plot and stan_plot_inferences 
  assign("rstan_chain_cols", rstancolc, e)
   
  # set the default number of parameters we are considered 
  # for plot of stanfit: when the number of parameters in a 
  # vector/array parameter is less than 
  # what is set here, we would have empty space. But when the 
  # number of parameters is larger than max, they are truncated. 
  assign('plot_standard_npar', 30, e)
  assign('plot_max_npar', 40, e)

  # color for shading the area of warmup trace plot
  assign("rstan_warmup_bg_col", rstan:::rstancolgrey[3], e)

  # boost lib path 
  rstan_inc_path  <- system.file('include', package = 'rstan')
  boost_lib_path <- file.path(rstan_inc_path, '/stanlib/boost_1.51.0') 
  eigen_lib_path <- system.file('include', package = 'RcppEigen')
  assign("eigen_lib", eigen_lib_path, e) 
  assign("boost_lib", boost_lib_path, e) 

  # cat("init_rstan_opt_env called.\n")
  invisible(e)
}

# init_rstan_opt_env(.rstan_opt_env)

rstan_options <- function(...) { 
  # Set/get options in RStan
  # Args: any options can be defined, using 'name = value' 
  #
  # e <- rstan:::.rstan_opt_env 
  e <- .rstan_opt_env 
  if (length(as.list(e)) == 0) 
    init_rstan_opt_env(e) 

  a <-  list(...)
  len <- length(a) 
  if (len < 1) return(NULL) 
  a_names <- names(a) 
  # deal with the case that this function is called as 
  # rstan_options("a", "b")
  if (is.null(a_names)) {
    ns <- unlist(a) 
    if (!is.character(ns)) 
      stop("rstan_options only accepts arguments as `name=value'") 

   ifnotfound_fun <- function(x) {
     warning("rstan option '", x, "' not found")
     NA 
   } 

    r <- mget(unlist(a), envir = e, ifnotfound = list(ifnotfound_fun)) 
    if (length(r) == 1) return(r[[1]])
    return(invisible(r))
  } 
  # the case for, for example, 
  # rstan_options(a = 3, b = 4, "c")
  empty <- (a_names == '') 

  opt_names <- c(a_names[!empty], unlist(a[empty]))
  r <- mget(opt_names, envir = e, ifnotfound = list(ifnotfound_fun))

  lapply(a_names[!empty], 
         FUN = function(n) {
           if (n == 'plot_rhat_breaks') {
             assign(n, sort(a[[n]]), e)
           } else {
             assign(n, a[[n]], e)
           }
         }) 

  if (length(r) == 1) return(r[[1]])
  invisible(r)
} 

## test code 
# o <- rstan_options() 
# o <- rstan_options(b = 4)
# ls(o)
