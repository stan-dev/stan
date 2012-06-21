
test.options1 <- function() {
  o <- as.list(rstan:::rstan.options()) 
  m <- match(c("plot.warmup.col", "plot.kept.col", "plot.chains.cols"), 
             names(o), 
             nomatch = 0) 
  checkTrue(all(m > 0)) 

  checkEquals(o[["plot.warmup.col"]], 19) 

  rstan:::rstan.options(plot.warmup.col = 22) 
  checkEquals(rstan:::get.rstan.options("plot.warmup.col"), 22) 

  rstan:::rstan.options(testname = 22) 
  checkEquals(rstan:::get.rstan.options("testname"), 22) 

  rstan:::rstan.options(a = 22, b = 23) 

} 

test.options2 <- function() {
  o <- rstan:::get.rstan.options(c('a', 'b')) 
  checkEquals(o$a, 22) 
  checkEquals(o$b, 23) 
} 
