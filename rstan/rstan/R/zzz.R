.onLoad <- function(libname, pkgname) {
    # require("methods", character = TRUE, quietly = TRUE)
    # loadRcppModules()
}

.onAttach <- function(...) {
  rstanLib <- dirname(system.file(package = "rstan"))
  pkgdesc <- packageDescription("rstan", lib.loc = rstanLib)
  builddate <- gsub(';.*$', '', pkgdesc$Packaged)
  gitrev <- substring(git_head(), 0, 12) 
  packageStartupMessage(paste("rstan (Version ", pkgdesc$Version, ", packaged: ", builddate, ", GitRev: ", gitrev, ")", sep = ""))
} 

