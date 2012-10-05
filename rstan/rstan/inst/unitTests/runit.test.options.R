
.setUp <- function() {
  rstan:::rstan_options(a = 22, b = 23) 
} 

test_options1 <- function() {
  o <- rstan:::rstan_options() 
  checkTrue(is.null(o)) 
  rstan:::rstan_options(testname = 22) 
  checkEquals(rstan:::rstan_options("testname"), 22) 
} 

test_options2 <- function() {
  o <- rstan:::rstan_options('a', 'b') 
  checkEquals(o$a, 22) 
  checkEquals(o$b, 23) 
  ov <- rstan:::rstan_options(a = 34)
  checkEquals(ov, 22) 
  o <- rstan:::rstan_options('a', 'b') 
  checkEquals(o$a, 34) 
  o <- rstan:::rstan_options('a', 'b', 'c') 
  checkEquals(o$c, NA) 
  o <- rstan:::rstan_options('a', 'b', 'c', d = 38) 
  checkEquals(o$d, NA) 
  checkEquals(rstan:::rstan_options("d"), 38)
} 

test_plot_rhat_breaks <- function() {
  rstan_options(plot_rhat_breaks = c(2, 1.2))
  o <- rstan_options("plot_rhat_breaks")
  checkEquals(o, c(1.2, 2)) 
  rstan_options(plot_rhat_breaks = c(1.5, 2, 1.2))
  o <- rstan_options("plot_rhat_breaks")
  checkEquals(o, c(1.2, 1.5, 2)) 
} 


# all options used in rstan 
test_rstan_options <- function() { 
  rhat_nan_col <- rstan_options("plot_rhat_nan_col")
  rhat_large_col <- rstan_options("plot_rhat_large_col")
  rhat_breaks <- rstan_options("plot_rhat_breaks")
  rhat_colors <- rstan_options("plot_rhat_cols")
  rhat_breaks <- rstan_options("plot_rhat_breaks")
  rhat_colors <- rstan_options("plot_rhat_cols")
  rhat_legend_cols <- c(rhat_colors, rstan_options('plot_rhat_large_col'),
                        rstan_options("plot_rhat_nan_col"))
  alert_col <- rstan_options("rstan_alert_col")
  chain_cols <- rstan_options("rstan_chain_cols")
  standard_width <- rstan_options('plot_standard_npar') 
  max_width <- rstan_options('plot_max_npar') 
  rstan_options("eigen_lib")
  rstan_options("boost_lib")
  rstan_options("rstan_chain_cols")
  warmup_col <- rstan_options("rstan_warmup_bg_col") 
} 

