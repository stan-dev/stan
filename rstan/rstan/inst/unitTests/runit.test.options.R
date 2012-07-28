
.setUp <- function() {
  rstan:::rstan.options(a = 22, b = 23) 
} 

test.options1 <- function() {
  o <- rstan:::rstan.options() 
  checkTrue(is.null(o)) 
  rstan:::rstan.options(testname = 22) 
  checkEquals(rstan:::rstan.options("testname"), 22) 
} 

test.options2 <- function() {
  o <- rstan:::rstan.options('a', 'b') 
  checkEquals(o$a, 22) 
  checkEquals(o$b, 23) 
  ov <- rstan:::rstan.options(a = 34)
  checkEquals(ov, 22) 
  o <- rstan:::rstan.options('a', 'b') 
  checkEquals(o$a, 34) 
} 
