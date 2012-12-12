
.setUp <- function() {
  model_code <- "model { \n y ~ normal(0, 1); \n}"  
  cat(model_code, file = 'tmp.stan')  

  a <- c(1, 3, 5)
  b <- matrix(1:10, ncol = 2)
  c <- array(1:18, dim = c(2, 3, 3)) 
  dump(c("a", "b", "c"), file = 'dumpabc.Rdump')
  rstan:::stan_rdump(c("a", "b", "c"), file = 'standumpabc.Rdump')

  cc <- c("# comment line 1", 
          " no comments line 1", 
          "# comment line 2",
          "# comment line 3", 
          "# comment line 4", 
          "# comment line 5", 
          " not comments line 2", 
          "# comment line 6",
          "not comments #comment line 7",
          "not comments at the end of file")
  cat(file = 'cc.csv', paste(cc, collapse = '\n'))
} 

test_get_model_strcode <- function() {
  model_code <- "model { \n y ~ normal(0, 1); \n}"  
  code <- 'parameters { real y; } model{y ~ normal(0,1);}'
  str1 <- rstan:::get_model_strcode("tmp.stan") 
  str2 <- rstan:::get_model_strcode(model_code = code)
  str3 <- rstan:::get_model_strcode(model_code = 'code') 
  str4 <- rstan:::get_model_strcode(model_code = 'parameters {real y;} model {y ~ normal(0,1); }') 
 
  mname1 <- attr(str1, 'model_name2')
  mname2 <- attr(str2, 'model_name2')
  mname3 <- attr(str3, 'model_name2')
  mname4 <- attr(str4, 'model_name2')
  checkEquals(mname1, 'tmp')
  checkEquals(mname2, 'code') 
  checkEquals(mname3, 'code')
  checkEquals(mname4, 'anon_model')
   
  attributes(str1) <- NULL 
  attributes(str2) <- NULL 
  attributes(str3) <- NULL 
  checkEquals(str1, model_code) 
  checkEquals(str2, code) 
  checkEquals(str3, code) 

  model_code <- "model { \n y ~ normal(0, 1); \n}"  
  # cat(model_code, file = 'tmp.stan')  
  checkEquals(model_code, rstan:::read_model_from_con('tmp.stan'), 
              msg = "Read stan model from file") 
  attr(model_code, 'model_name2') <- 'tmp' 
  checkEquals(model_code, rstan:::get_model_strcode('tmp.stan'), 
              msg = "Read stan model from file") 
  attr(model_code, 'model_name2') <- 'model_code' 
  checkEquals(model_code, rstan:::get_model_strcode(model_code = model_code), 
              msg = "Read stan model from model_code") 
  checkException(rstan:::get_model_strcode(), 
                 msg = "Read stan model from model_code") 
} 

test_is_valid_stan_name <- function() {
  checkTrue(!rstan:::is_legal_stan_vname('7dd'))
  checkTrue(!rstan:::is_legal_stan_vname('model'))
  checkTrue(!rstan:::is_legal_stan_vname('private'))
  checkTrue(!rstan:::is_legal_stan_vname('hello__'))
  checkTrue(rstan:::is_legal_stan_vname('y'))
} 

test_data_preprocess <- function() {
  lst <- list(z = c(1L, 2L, 4L), 
              a = 1:100, 
              b = matrix(1:9 / 9, ncol = 3), 
              c = structure(1:100, .Dim = c(5, 20)),
              g = array(c(3, 3, 9, 3, 3, 4, 5, 6, 9, 8, 0, 2), dim = c(2, 2, 3)), 
              d = 1:100 + .1) 
  lst <- rstan:::data_preprocess(lst) 
  lst2 <- lst  
  lst2$f <- matrix(c(3, NA, NA, NA, 3, 4), ncol = 3) 
  lst3 <- lst
  lst3$h <- gl(3, 4)
  lst4 <- rstan:::data_preprocess(lst3)

  checkEquals(dim(lst$g), c(2, 2, 3), "Keep the dimension information")
  checkTrue(is.integer(lst$z), "Do as.integer when appropriate") 
  checkTrue(is.double(lst$b), msg = "Not do as.integer when it is not appropriate") 
  checkException(rstan:::data_preprocess(lst2), 
                 msg = "Stop if data have NA") 
  checkEquals(names(lst4), c("z", "a", "b", "c", "g", "d")) # check if h is removed

} 

test_data_preprocess2 <- function() {
  # a list of array as an element of the data list 
  I <- 3; J <- 4; K <- 5
  a <- lapply(1:I, function(i) rnorm(J))
  b <- lapply(1:I, function(i) matrix(rnorm(J * K), ncol = K))
  d <- lapply(1:I, function(i) rnorm(1, i))
  e <- lapply(1:I, function(i) rpois(J, 1) + 1.0) 
  lst2 <- rstan:::data_preprocess(list(a = a, b = b, d = d, e = e))
  checkEquals(dim(lst2$a), c(I, J))
  checkEquals(dim(lst2$b), c(I, J, K))
  checkEquals(dim(lst2$d), c(I, 1))
  checkTrue(is.integer(lst2$e[1, 1])) 
  checkTrue(!is.integer(e[[1]][1])) 

  a1 <- lapply(1:I, function(i) rnorm(J))
  a2 <- lapply(1:I, function(i) rnorm(J))
  checkException(rstan:::data_preprocess(list(a = list(a1 = a1, a2 = a2))))
} 


test_read_rdump <- function() {
  l <- rstan:::read_rdump("dumpabc.Rdump")
  checkEquals(l$a, c(1, 3, 5)) 
  checkEquals(l$b, matrix(1:10, ncol = 2))
  checkEquals(l$c, array(1:18, dim = c(2, 3, 3))) 
} 

test_stan_rdump <- function() {
  l <- rstan:::read_rdump("standumpabc.Rdump")
  checkEquals(l$a, c(1, 3, 5)) 
  checkEquals(l$b, matrix(1:10, ncol = 2))
  checkEquals(l$c, array(1:18, dim = c(2, 3, 3))) 
} 

test_seq_array_ind <- function() {
  a <- rstan:::seq_array_ind(numeric(0))
  checkEquals(length(a), 0) 
  # by default, col_major is FALSE
  b <- rstan:::seq_array_ind(2:5, col_major = TRUE) 
  c <- arrayInd(1:prod(2:5), .dim = 2:5) 
  checkEquals(b, c) 
  d <- rstan:::seq_array_ind(2:3, col_major = FALSE)
  e <- matrix(c(1, 1, 1, 2, 1, 3, 2, 1, 2, 2, 2, 3), 
              nrow = 6, byrow = TRUE)
  checkEquals(d, as.array(e)) 
} 

test_flatnames <- function() {
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

test_idx_col2rowm <- function() {
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

test_idx_row2colm <- function() {
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

test_pars_total_indexes <- function() {
  names <- "alpha0" 
  dims <- list(alpha0 = c(2, 3)) 
  fnames <- rstan:::flatnames(names, dims)  
  tidx <- rstan:::pars_total_indexes(names, dims, fnames, "alpha0") 
  tidx.attr1 <- attr(tidx[[1]], "row_major_idx") 
  attributes(tidx[[1]]) <- NULL 
  checkEquals(unname(tidx[[1]]), 1:6) 
  checkEquals(unname(tidx.attr1), c(1, 3, 5, 2, 4, 6)) 
  names2 <- c(names, "alpha") 
  dims2 <- c(dims, list(alpha = 8)) 
  fnames2 <- rstan:::flatnames(names2, dims2)  
  tidx2 <- rstan:::pars_total_indexes(names2, dims2, fnames2, "alpha") 
  tidx2.attr1 <- attr(tidx2[[1]], "row_major_idx")
  attributes(tidx2[[1]]) <- NULL
  checkEquals(unname(tidx2[[1]]), 6 + 1:8)  
  checkEquals(unname(tidx2.attr1), 6 + 1:8)
  names3 <- c(names2, "p") 
  dims3 <- c(dims2, list(p = integer(0)))
  fnames3 <- rstan:::flatnames(names3, dims3)  
  tidx3 <- rstan:::pars_total_indexes(names3, dims3, fnames3, "p") 
  tidx3.attr1 <- attr(tidx3[[1]], "row_major_idx")
  attributes(tidx3[[1]]) <- NULL
  checkEquals(unname(tidx3[[1]]), 15)  
  checkEquals(unname(tidx3.attr1), 15)
} 

test_multi_idx_row2colm <- function() {
  checkEquals(rstan:::multi_idx_row2colm(list(integer(0))), 1)
  dims <- list(c(3), c(2,3), integer(0), c(2))
  col_idx <- rstan:::multi_idx_row2colm(dims)
  target <- c(1, 2, 3, 4, 7, 5, 8, 6, 9, 10, 11, 12)
  checkEquals(col_idx, target)

  fnames <- c("alpha[1]", "alpha[2]", "alpha[3]", 
              "alpha2[1,1]", "alpha2[1,2]", "alpha2[1,3]", 
              "alpha2[2,1]", "alpha2[2,2]", "alpha2[2,3]", 
              "p", "theta[1]", "theta[2]")
  fnames_colm <- c("alpha[1]", "alpha[2]", "alpha[3]", 
                   "alpha2[1,1]", "alpha2[2,1]", "alpha2[1,2]", "alpha2[2,2]", 
                   "alpha2[1,3]", "alpha2[2,3]", "p", "theta[1]", "theta[2]")
  checkEquals(fnames[col_idx], fnames_colm)
} 

test_mklist <- function() {
  x <- 3:5 
  y <- array(1:9, dim = c(3, 3))  
  z <- list(p = 3)
  f <- function() { TRUE }
  a <- list(x = x, y = y) 
  b <- rstan:::mklist(c("x", "y")) 
  checkTrue(identical(a, b)) 
  c <- list(x = x, y = y, z = z) 
  d <- rstan:::mklist(c("x", "y", "z"))
  checkTrue(identical(c, d))
  checkException(rstan:::mklist(c("x", "f")))
} 

test_makeconf_path <- function() {
  p <- makeconf_path()  
  checkTrue(file.exists(makeconf_path()))
} 

test_config_argss <- function() {
  # (chains, iter, warmup, thin, init, seed, sample_file, ...)
  a <- rstan:::config_argss(3, 100, 10, 3, 0, 0, "a.csv", chain_id = 4)
  checkEquals(length(a), 3)
  checkEquals(a[[1]]$init, "0") 
  checkEquals(a[[1]]$chain_id, 4) 
  checkEquals(a[[3]]$chain_id, 6) 
  b <- rstan:::config_argss(3, 100, 10, 3, "0", 10, "a.csv") 
  checkEquals(b[[3]]$chain_id, 3) 
  checkEquals(b[[1]]$init, "0") 
  c <- rstan:::config_argss(3, 100, 10, 3, "random", 10, "a.csv") 
  checkEquals(c[[1]]$init, "random") 
  d <- rstan:::config_argss(4, 100, 10, 3, "random", 10, "a.csv", chain_id = c(3, 2, 1)) 
  checkEquals(d[[3]]$chain_id, 1) 
  checkEquals(d[[4]]$chain_id, 4) 
  checkException(rstan:::config_argss(3, 100, 10, 3, "random", 10, NA, chain_id = c(3, 3)))
  b <- rstan:::config_argss(3, 100, 10, 3, 0, "12345", "a.csv", chain_id = 4)
  checkEquals(b[[1]]$seed, '12345')
  checkException(rstan:::config_argss(3, 100, 10, 3, 0, "a12345", "a.csv", chain_id = 4))
  checkException(rstan:::config_argss(3, 100, 10, 3, 0, "1a2345", "a.csv", chain_id = 4))
} 

test_data_list2array <- function() {
  d <- list(y = rnorm(20))
  d2 <- rstan:::data_list2array(d)
  checkEquals(d2, array(d$y, dim = c(1, 20)))

  I <- 4; J <- 5; K <- 6
  b <- lapply(1:I, function(i) rnorm(J))
  b2 <- rstan:::data_list2array(b)
  b3 <- data.matrix(do.call(rbind, b))
  checkEquals(b2, b3)

  a <- lapply(1:I, function(i) array(rnorm(J * K), dim = c(J, K)))
  a2 <- rstan:::data_list2array(a)
  for (i in 1:I) checkEquals(a[[i]], a2[i,,])
  checkEquals(a[[4]][5, 6], a2[4, 5, 6]) 
  checkEquals(a[[3]][4, 2], a2[3, 4, 2]) 
  checkEquals(a[[2]][1, 6], a2[2, 1, 6]) 
}

test_read_comments <- function() {
  a1 <- rstan:::read_comments('cc.csv', 5L)
  checkEquals(a1[1], "# comment line 1")
  checkEquals(a1[5], "# comment line 5")
  a2 <- rstan:::read_comments('cc.csv', 3L)
  checkEquals(length(a2), 3L)
  a3 <- rstan:::read_comments('cc.csv', -1)
  checkEquals(length(a3), 7L)
} 

test_get_dims_from_fnames <- function() {
  names <- c('alpha', 'beta2', 'g2amma', 'theta0')
  dims <- list(c(2L, 3L), integer(0L), c(4L), c(3L, 5L, 4L))
  fnames <- rstan:::flatnames(names, dims)
  fnames_d <- rstan:::sqrfnames_to_dotfnames(fnames)
  unames <- rstan:::unique_par(fnames_d)
  dims2 <- lapply(unames, 
                  function(n) { 
                    fnames_d2 <- fnames_d[sapply(fnames_d, function(i) grepl(n, i))]
                    # the above line works here since all parameters are not nested. 
                    # it would be probematic if say we have another parameter `p`,
                    # since p is also part of `alpha`
                    rstan:::get_dims_from_fnames(fnames_d2)
                  })
  checkEquals(dims, dims2)
} 


test_dotfnames_fromto_sqrfnames <- function() {
  dn <- c("alpha", "beta.1", "beta.2", "gamma.1.2", "gamma.1.4")
  sn <- rstan:::dotfnames_to_sqrfnames(dn)
  dn2 <- rstan:::sqrfnames_to_dotfnames(sn)
  checkEquals(dn, dn2)
}

test_par_vector2list <- function() {
  v <- c(2.3, 3.4, 4.5, (1:8)/9); 
  pars <- c('alpha', 'beta', 'gamma')
  dims <- list(integer(0), c(2), c(2, 4))
  vl <- rstan:::par_vector2list(v, pars, dims)
  alpha <- 2.3  
  beta <- array(v[2:3], dim = 2) 
  gamma <- array(v[4:11], dim = c(2, 4))
  checkEquals(length(vl), 3)
  checkEquals(vl[[1]], alpha)
  checkEquals(vl[[2]], beta)
  checkEquals(vl[[3]], gamma)
} 
 
.tearDown <- function() {
  unlink('tmp.stan') 
  unlink('dumpabc.Rdump') 
  unlink('standumpabc.Rdump') 
  unlink('cc.csv')
}

