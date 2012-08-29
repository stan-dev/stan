
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
} 
