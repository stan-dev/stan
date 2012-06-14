
#   is.whole.number <- function(x) {
#     all.equal(x, round(x), check.attributes = FALSE) 
#   } 

#   as.integer.if.doable <- function(y) {
#     if (!is.numeric(y)) return(y) 
#     if (is.integer(y)) return(y) 
#     if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
#       storage.mode(y) <- "integer"  
#     return(y) 
#   } 


## @param x is a list 
## 
## Ignore non-numeric vectors since we ignore
## them in rlist_var_context 
## 
list.as.integer.if.doable <- function(x) {
  lapply(x, 
         FUN = function(y) { 
           if (!is.numeric(y)) return(y) 
           if (is.integer(y)) return(y) 
           if (isTRUE(all.equal(y, round(y), check.attributes = FALSE))) 
             storage.mode(y) <- "integer"  
           return(y) 
         });  
} 

## Preprocess the data (list or env) to list for stan
## @param data A list or environment: 
##  1 stop if there is NA; no-name lists; duplicate names  
##  2 remove NULL, non-numeric elements 
##  3 change to integers when applicable 

data.preprocess <- function(data) { # , varnames) {

  # 
  # if (is.environment(data)) {
    
  #   data <- mget(varnames, envir = data, mode = "numeric", 
  #                ifnotfound = list(NULL))
  #   data <- data[!sapply(data, is.null)]
  # }
  if (is.environment(data)) {
    data <- as.list(data) 
  } else if (is.list(data)) {
    v <- names(data)
    if (is.null(v)) 
      stop("data must be a named list")
          
    ## Stan would report error if variable is not found 
    ## from the list
    # if (any(nchar(v) == 0))  
    #   stop("unnamed variables in data list")
    # 
 
    if (any(duplicated(v))) {
      stop("Duplicated names in data list: ", 
           paste(v[duplicated(v)], collapse = " "))
    }
  } else {
    stop("data must be a list or environment")
  } 
 
  data <- lapply(data, 
                 FUN = function(x) {
 
                   ## Now we stop whenever we have NA in the data
                   ## since we do not know what variables are needed
                   ## at this pointi.
                   if (any(is.na(x))) {
                     stop("Stan does not support NA in the data.\n");
                   } 
 
                   # remove those not numeric data 
                   if (!is.numeric(x)) return(NULL) 
 
                   if (is.integer(x)) return(x) 
         
                   # change those integers in form of real to integers 
                   if (isTRUE(all.equal(x, round(x), check.attributes = FALSE))) 
                     storage.mode(x) <- "integer"  
                   return(x) 
                 })   
 
  data[!sapply(data, is.null)] 
} 


read.model.from.con <- function(con) {
  lines <- readLines(con, n = -1L, warn = FALSE);
  paste(lines, collapse = '\n') 
} 

get.model.code <- function(file, model.code = '') {
  if (!missing(file)) {
    if (is.character(file)) {
      fname <- file
      file <- try(file(fname, "rt"))
      if (inherits(file, "try-error")) {
        stop(paste("Cannot open model file \"", fname, "\"", sep = ""))
      }
      on.exit(close(file))
    } else if (!inherits(file, "connection")) {
      stop("'File' must be a character string or connection")
    }
    model.code <- paste(readLines(file, warn = FALSE), collapse = '\n') 
  } else if (model.code == '') {  
    stop("Missing model file missing and empty model.code")
  } 
  model.code 
} 



# FIXEME: implement more check on the arguments 
check.args <- function(argss) {
  if (FALSE) stop() 
} 

#
# model.code <- read.model.from.con('http://stan.googlecode.com/git/src/models/bugs_examples/vol1/dyes/dyes.stan');
# cat(model.code)


append.id <- function(file, id, suffix = '.csv') {
  fname <- basename(file)
  fpath <- dirname(file)
  fname2 <- gsub("\\.csv[[:space:]]*$", 
                 paste("_", id, ".csv", sep = ''), 
                 fname)
  if (fname2 == fname) 
    fname2 <- paste(fname, "_", id, ".csv", sep = '')
  file.path(fpath, fname2)
}



config.argss <- function(n.chains, n.iter, n.warmup, n.thin, 
                        init.t, init.v,
                        seed, sample.file, ...) {

  ## seed: only one seed is needed by virtue of the RNG 

  n.iters <- rep(n.iter, n.chains)   
  n.thins <- rep(n.thin, n.chains)  
  n.warmups <- rep(n.warmup, n.chains) 

  init.t <- as.character(init.t)
  init.t[which(!init.t %in% c("0", "user"))] <- 'random'

  init.ts <- rep(init.t, n.chains)  
  
  init.vs <- vector("list", n.chains) 

  if (!missing(init.v) && !is.null(init.v)) {
    if (is.function(init.v)) {
      ## the function could take an argument named by chain.id 
      ## from 1 to num.chains. 
      if (any(names(formals(init.v)) == "chain.id")) {
        for (i in 1:n.chains)  
          init.vs[[i]] <- init.v(chain.id = i)
      } else {
        for (i in 1:n.chains)  
          init.vs[[i]] <- init.v() 
      } 
    } else if (is.list(init.v)) {
      if (length(init.v) != n.chains) 
        stop("Initial value list mismatch number of chains") 
      if (!any(sapply(init.v, is.list))) {
        # print(init.v)
        stop("Initial value list is not a list of lists") 
      }
      init.vs <- init.v 
    } else { 
        stop("Wrong specification of initial values")
    } 
  } 

  argss <- vector("list", n.chains)  
  ## the name of arguments in the list need to 
  ## match those in include/rstan/stan_args.hpp 
  for (i in 1:n.chains)  
    argss[[i]] <- list(chain_id = i, 
                       iter = n.iters[i], thin = n.thins[i], 
                       warmup = n.warmups[i], init = init.ts[i]) 
                
    
  if (!missing(init.v) && !is.null(init.v))  
    for (i in 1:n.chains) 
      argss[[i]]$init_list = init.vs[[i]]   

  if (!missing(seed))  
      argss[[i]]$seed <- seed; 

  if (!missing(sample.file)) {
    if (n.chains == 1) 
        argss[[1]]$sample_file <- sample.file
    if (n.chains > 1) {
      for (i in 1:n.chains) 
        argss[[i]]$sample_file <- append.id(sample.file, i) 
    }
  }

  check.args(argss) 
  
  argss 
} 


probs2str <- function(probs) {
  paste(formatC(probs * 100,  
                digits = 1, 
                format = 'f', 
                drop0trailing = TRUE), 
        "%", sep = '')
} 

stan.dump <- function(data, file) {
  # Dump an R list or environment for a model data 
  # to the R dump file that Stan supports.
  #
  # Args:
  #   data: the data, an object of list of environment.
  #   file: the output file for dumping the variables. 
  # 
  # Retrun:
 
  if (missing(data)) 
    stop("error: stan.dump needs argument 'data'") 
  if (missing(file)) 
    stop("error: stan.dump needs argument 'file', ",
         "into which the data are dumped.") 

  ### FIXEME, to be implemented. 
} 


#### temporary test code 
#  a <- config.argss(3, c(100, 200), 10, 1, "user", NULL, seed = 3) 
#  print(a) 
#  
#  fun1 <- function(chain.id) {
#    cat("chain.id=", chain.id)
#    return(list(mu = chain.id));
#  } 
#  b <- config.argss(3, c(100, 200), 10, 1, c("user", 1), fun1, seed = 3) 
#  print(b)
#  
#  
