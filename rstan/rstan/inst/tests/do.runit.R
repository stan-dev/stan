library(rstan) 
library(RUnit) 
rstantest <- defineTestSuite("rstantest",
                             dirs = file.path(.path.package(package="rstan"), "tests"),
                             testFileRegexp = "^runit.+\\.R",
                             testFuncRegexp = "^test.+",
                             rngKind = "Marsaglia-Multicarry",
                             rngNormalKind = "Kinderman-Ramage")


testResult <- runTestSuite(rstantest) 
printTextProtocol(testResult, showDetails = TRUE)  
# printHTMLProtocol(testResult, fileName = 'testo.html')


#   checkEquals(target, current, msg),
#   checkEqualsNumeric(target, current, msg)
#   checkIdentical(target, current, msg)
#   checkTrue(expr, msg)
#   checkException(expr, msg, silent = getOption("RUnit")$silent)
