
.setUp <- function() {
  model.code <- "model { \n y ~ normal(0, 1); \n}"  
  cat(model.code, file = 'tmp.stan')  

  a <- c(1, 3, 5)
  b <- matrix(1:10, ncol = 2)
  c <- array(1:18, dim = c(2, 3, 3)) 
  dump(c("a", "b", "c"), file = 'dumpabc.Rdump')
  rstan:::stan.rdump(c("a", "b", "c"), file = 'standumpabc.Rdump')
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
  l <- rstan:::read.rdump("dumpabc.Rdump")
  checkEquals(l$a, c(1, 3, 5)) 
  checkEquals(l$b, matrix(1:10, ncol = 2))
  checkEquals(l$c, array(1:18, dim = c(2, 3, 3))) 
} 

test.stan.rdump <- function() {
  l <- rstan:::read.rdump("standumpabc.Rdump")
  checkEquals(l$a, c(1, 3, 5)) 
  checkEquals(l$b, matrix(1:10, ncol = 2))
  checkEquals(l$c, array(1:18, dim = c(2, 3, 3))) 
} 

test.seq.array.ind <- function() {
  a <- rstan:::seq.array.ind(numeric(0))
  checkEquals(length(a), 0) 
  # by default, col.major is FALSE
  b <- rstan:::seq.array.ind(2:5, col.major = TRUE) 
  c <- arrayInd(1:prod(2:5), .dim = 2:5) 
  checkEquals(b, c) 
  d <- rstan:::seq.array.ind(2:3, col.major = FALSE)
  e <- matrix(c(1, 1, 1, 2, 1, 3, 2, 1, 2, 2, 2, 3), 
              nrow = 6, byrow = TRUE)
  checkEquals(d, as.array(e)) 
} 

test.flatnames <- function() {
  names <- c("alpha", "beta", "gamma", "delta") 
  dims <- list(alpha = integer(0), beta = c(2, 3), gamma = c(2, 3, 4), delta = c(5))
  fnames <- rstan:::flatnames(names, dims)  
  checkEquals(fnames, 
             c('alpha', "beta[1,1]", "beta[1,2]", "beta[1,3]", 
                        "beta[2,1]", "beta[2,2]", "beta[2,3]", 
                        "gamma[1,1,1]", "gamma[1,1,2]", "gamma[1,1,3]", "gamma[1,1,4]", 
                        "gamma[1,2,1]", "gamma[1,2,2]", "gamma[1,2,3]", "gamma[1,2,4]", 
                        "gamma[1,3,1]", "gamma[1,3,2]", "gamma[1,3,3]", "gamma[1,3,4]", 
                        "gamma[2,1,1]", "gamma[2,1,2]", "gamma[2,1,3]", "gamma[2,1,4]", 
                        "gamma[2,2,1]", "gamma[2,2,2]", "gamma[2,2,3]", "gamma[2,2,4]", 
                        "gamma[2,3,1]", "gamma[2,3,2]", "gamma[2,3,3]", "gamma[2,3,4]", 
                        "delta[1]", "delta[2]", "delta[3]", "delta[4]", "delta[5]")) 
  names2 <- c('alpha') 
  dims2 <- list(alpha = integer(0))
  fnames2 <- rstan:::flatnames(names2, dims2)  
  checkEquals(fnames2, "alpha")
} 

test.idx_col2rowm <- function() {
  d <- integer(0) 
  idx <- rstan:::idx_col2rowm(d) 
  checkEquals(idx, 1) 
  d2 <- 8 
  idx2 <- rstan:::idx_col2rowm(d2) 
  checkEquals(idx2, 1:8) 
  d3 <- c(3, 4, 5) 
  idx3 <- rstan:::idx_col2rowm(d3) 
  yidx3 <- c(1, 13, 25, 37, 49, 4, 16, 28, 40, 52, 7, 19, 31, 43, 55,
             10, 22, 34, 46, 58, 2, 14, 26, 38, 50, 5, 17, 29, 41, 53, 8, 20, 32, 44, 56,
             11, 23, 35, 47, 59, 3, 15, 27, 39, 51, 6, 18, 30, 42, 54, 9, 21, 33, 45, 57,
             12, 24, 36, 48, 60) 
  checkEquals(idx3, yidx3)
} 

test.idx_row2colm <- function() {
  d <- integer(0) 
  idx <- rstan:::idx_row2colm(d) 
  checkEquals(idx, 1) 
  d2 <- 8 
  idx2 <- rstan:::idx_row2colm(d2) 
  checkEquals(idx2, 1:8) 
  d3 <- c(3, 4, 5) 
  idx3 <- rstan:::idx_row2colm(d3) 
  yidx3 <- c(1, 21, 41, 6, 26, 46, 11, 31, 51, 16, 36, 56, 2, 22, 42, 7, 27, 47, 12, 32, 52,
             17, 37, 57, 3, 23, 43, 8, 28, 48, 13, 33, 53, 18, 38, 58, 4, 24, 44, 9, 29, 49,
             14, 34, 54, 19, 39, 59, 5, 25, 45, 10, 30, 50, 15, 35, 55, 20, 40, 60) 
  checkEquals(idx3, yidx3)
} 

test.pars.total.indexes <- function() {
  names <- "alpha" 
  dims <- list(alpha = c(2, 3)) 
  fnames <- rstan:::flatnames(names, dims)  
  tidx <- rstan:::pars.total.indexes(names, dims, fnames, "alpha") 
  tidx.attr1 <- attr(tidx[[1]], "row.major.idx") 
  attributes(tidx[[1]]) <- NULL 
  checkEquals(unname(tidx[[1]]), 1:6) 
  checkEquals(unname(tidx.attr1), c(1, 3, 5, 2, 4, 6)) 
  names2 <- c(names, "beta") 
  dims2 <- list(alpha = c(2, 3), beta = 8) 
  fnames2 <- rstan:::flatnames(names2, dims2)  
  tidx2 <- rstan:::pars.total.indexes(names2, dims2, fnames2, "beta") 
  tidx2.attr1 <- attr(tidx2[[1]], "row.major.idx")
  attributes(tidx2[[1]]) <- NULL
  checkEquals(unname(tidx2[[1]]), 6 + 1:8)  
  checkEquals(unname(tidx2.attr1), 6 + 1:8)
} 

test.mklist <- function() {
  x <- 3:5 
  y <- array(1:9, dim = c(3, 3))  
  assign("x", x, envir = .GlobalEnv) 
  assign("y", y, envir = .GlobalEnv) 
  a <- list(x = x, y = y) 
  b <- rstan:::mklist(c("x", "y")) 
  checkTrue(identical(a, b)) 
} 

test.makeconf.path <- function() {
  p <- makeconf.path()  
  checkTrue(file.exists(makeconf.path()))
} 
 
.tearDown <- function() {
  unlink('tmp.stan') 
  unlink('dumpabc.Rdump') 
  unlink('standumpabc.Rdump') 
} 

