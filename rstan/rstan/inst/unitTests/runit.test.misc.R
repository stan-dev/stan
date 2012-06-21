
.setUp <- function() {
  model.code <- "model { \n y ~ normal(0, 1); \n}"  
  cat(model.code, file = 'tmp.stan')  

  a <- c(1, 3, 5)
  b <- matrix(1:10, ncol = 2)
  c <- array(1:18, dim = c(2, 3, 3)) 
  dump(c("a", "b", "c"), file = 'dumpabc.R')
} 


test.util <- function() {
  lst <- list(z = c(1L, 2L, 4L), 
              a = 1:100, 
              b = matrix(1:9 / 9, ncol = 3), 
              c = structure(1:100, .Dim = c(5, 20)),
              g = array(c(3, 3, 9, 3, 3, 4, 5, 6, 9, 8, 0, 2), dim = c(2, 2, 3)), 
              d = 1:100 + .1) 
  lst <- rstan:::data.preprocess(lst) 
  lst2 <- lst; 
  lst2$f <- matrix(c(3, NA, NA, NA, 3, 4), ncol = 3) 

  checkEquals(dim(lst$g), c(2, 2, 3), "Keep the dimension infomation")
  checkTrue(is.integer(lst$z), "Do as.integer when appropriate") 
  checkTrue(is.double(lst$b), msg = "Not do as.integer when it is not appropriate") 
  checkException(rstan:::data.preprocess(lst2), 
                 msg = "Stop if data have NA") 

  model.code <- "model { \n y ~ normal(0, 1); \n}"  
  # cat(model.code, file = 'tmp.stan')  
  checkEquals(model.code, rstan:::read.model.from.con('tmp.stan'), 
              msg = "Read stan model from file") 
  checkEquals(model.code, rstan:::get.model.code('tmp.stan'), 
              msg = "Read stan model from file") 
  checkEquals(model.code, rstan:::get.model.code(model.code = model.code), 
              msg = "Read stan model from model.code") 
  checkException(rstan:::get.model.code(), 
                 msg = "Read stan model from model.code") 
} 


test.read.rdump <- function() {
  l <- rstan:::read.rdump("dumpabc.R")
  checkEquals(l$a, c(1, 3, 5)) 
  checkEquals(l$b, matrix(1:10, ncol = 2))
  checkEquals(l$c, array(1:18, dim = c(2, 3, 3))) 
} 

.tearDown <- function() {
  unlink('tmp.stan') 
  unlink('dumpabc.R') 
} 

