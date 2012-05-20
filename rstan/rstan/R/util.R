
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


#   print(lst1)
#   list.data.preprocess(lst1) 
#   print(lst1)

#   lst1$e <- "hello"
#   list.data.preprocess(lst1) 
#   lst1$f <- c(3, NA) 
#   lst1$f <- matrix(c(3, NA, NA, NA, 3, 4), ncol = 3) 
#   lst1$f <- array(c(3, NA, NA, NA, 3, 4, 5, 6, 9, 8, NA, 2), dim = c(2, 2, 3)) 
#   list.data.preprocess(lst1) 

