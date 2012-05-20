

test.util <- function() {
    lst <- list(z = c(1L, 2L, 4L), 
                a = 1:100, 
                b = matrix(1:9 / 9, ncol = 3), 
                c = structure(1:100, .Dim = c(5, 20)),
                g = array(c(3, 3, 9, 3, 3, 4, 5, 6, 9, 8, 0, 2), dim = c(2, 2, 3)), 
                d = 1:100 + .1) 
    lst <- list.data.preprocess(lst) 
    lst2 <- lst; 
    lst2$f <- matrix(c(3, NA, NA, NA, 3, 4), ncol = 3) 

    checkEquals(dim(lst$g), c(2, 2, 3), "Keep the dimension infomation")
    checkTRUE(is.integer(lst$z), "Do as.integer when appropriate") 
    checkTRUE(is.double(lst$b), "Not do as.integer when it is not appropriate") 
    checkException(list.data.preprocess(lst2), msg = "Stop if data have NA") 

} 


