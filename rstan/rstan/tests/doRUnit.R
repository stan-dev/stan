
## unit tests will not be done if RUnit is not available
if(!require("RUnit", quietly = TRUE)) {
  warning("cannot run unit tests -- package RUnit is not available")
  q('no');
} 

 
pkg <- "rstan" 

path <- system.file(package = pkg, "unitTests")




source(file.path(path, "runTests.R"), echo = TRUE)
 

