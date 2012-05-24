STAN_HOME <- Sys.getenv('STAN_HOME')
PKG_CPPFLAGS_env <- paste(" -I", paste(STAN_HOME, '/lib', sep = ''), 
                          " -I", paste(STAN_HOME, '/src', sep = ''), 
                          " -I", system.file('include', package = 'rstan')) 

rstanplugin <- function() {
  Rcpp_plugin <- getPlugin("Rcpp") 
  list(includes = '', 
       body = function(x) x, 
       LinkingTo = c("Rcpp"),
       env = list(PKG_LIBS = paste(Rcpp_plugin$env$PKG_LIBS, " -L ", STAN_HOME, "/bin -lstan", sep = ''), 
                  PKG_CPPFLAGS = paste(Rcpp_plugin$env$PKG_CPPFLAGS, PKG_CPPFLAGS_env)))  
} 

# registerPlugin("rstan", rstanplugin); 

inlineCxxPlugin <- function(...) {
  settings <- rstanplugin()  
  settings
}

appendLDLIBPATH <- function(x) {
  ldlpath <- Sys.getenv("LD_LIBRARY_PATH") 
  ldlpath <- paste(ldlpath, ":", x, sep = '') 
  ## 
  ## assuming unix-like system for the time being 
  ## FIXME later 
  Sys.setenv(LD_LIBRARY_PATH=ldlpath); 
} 
