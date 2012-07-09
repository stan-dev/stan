require(Rcpp)
require(inline)
require(rstan) 

inc <- paste(readLines('test_rlist_var_context.cpp'), collapse = '\n')

fx <- cxxfunction(signature(), "" , includes = inc, 
                  # settings = myplugin, verbose = TRUE); 
                  plugin = "rstan", verbose = TRUE)
mod <- Module("rstantest", getDynLib(fx))

lst1 <- list(foo = c(1L, 2L)); 
lst2 <- list(foo = c(1L, 2L), bar = 1.0) 
lst3 <- list(foo = c(1L, 2L),
             bar = 1L, 
             bing = structure(c(1.0, 2.0, 2.0, 5.0, 3.0, 6.0), .Dim = c(2, 3))); 

mod$test_rlist_var_context1(lst1); 
mod$test_rlist_var_context2(lst2); 
mod$test_rlist_var_context3(lst3);

