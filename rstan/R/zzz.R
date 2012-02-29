
.onLoad <- function(pkgname, libname){
    require("methods", character=TRUE, quietly=TRUE)
    loadRcppModules()
}

