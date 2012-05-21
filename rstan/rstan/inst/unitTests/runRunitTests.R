pkg <- 'rstan' 

if(!require("RUnit", quietly = TRUE)) {
  stop("Package Runit is not found.") 
} 

if (exists("path")) { 
  reportfile <- file.path(getwd(), "report")
} else {
   path <- getwd() 
   reportfile <- file.path(path, "report") 
} 

stopifnot(file.exists(path), file.info(path.expand(path))$isdir)
library(package = pkg, character.only = TRUE)

rstantest <- defineTestSuite("rstantest",
                             dirs = path, 
                             testFileRegexp = "^runit.+\\.R",
                             testFuncRegexp = "^test.+",
                             rngKind = "Marsaglia-Multicarry",
                             rngNormalKind = "Kinderman-Ramage")

testsres <- runTestSuite(rstantest) 
 

printTextProtocol(testsres, showDetails = TRUE)
printTextProtocol(testsres, showDetails = TRUE,
                  fileName = paste(reportfile, ".txt", sep=""))
 
printHTMLProtocol(testsres, 
                  fileName = paste(reportfile, ".html", sep = ""))
 
tmp <- getErrors(testsres)
if(tmp$nFail > 0 | tmp$nErr > 0) {
  stop(paste("\n\nUnit testing failed (#test failures: ", tmp$nFail,
             ", #R errors: ",  tmp$nErr, ")\n\n", sep = ""))
} 
