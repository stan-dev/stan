
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


# x is a list 
# ignore non-numeric vectors since we ignore
# them in rlist_var_context 
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

list.data.preprocess <- function(x) {
  lapply(x, 
         FUN = function(y) {
           if (any(is.na(y))) {
             stop("Stan does not support NA in the data.\n");
           } 
         });  
  list.as.integer.if.doable(x) 
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


#
# model.code <- read.model.from.con('http://stan.googlecode.com/git/src/models/bugs_examples/vol1/dyes/dyes.stan');
# cat(model.code)


