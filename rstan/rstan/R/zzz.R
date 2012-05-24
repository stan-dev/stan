
.onLoad <- function(libname, pkgname) {
    # require("methods", character = TRUE, quietly = TRUE)

    ## not working since this function is not executed before loading 
    ## 
    # STAN_HOME <- Sys.getenv('STAN_HOME')
    # appendLDLIBPATH(paste(Sys.getenv('STAN_HOME'), "/bin", sep = '')) 
    
    loadRcppModules()
}

