rstan.inc.path  <- system.file('include', package = 'rstan')
rstan.libs.path <- system.file('libstan', package = 'rstan')

# Using RcppEigen 
# eigen.path <- system.file('include', package = 'RcppEigen')

# If included in RStan 
eigen.path <- paste0(rstan.inc.path, '/stanlib/eigen_3.1.0') 

PKG_CPPFLAGS_env <- paste0(' -I"', paste0(rstan.inc.path, '/stansrc" '), 
                           ' -I"', paste0(eigen.path, '" '), 
                           ' -I"', paste0(eigen.path, '/unsupported" '), 
                           ' -I"', paste0(rstan.inc.path, '/stanlib/boost_1.50.0" '), 
                           ' -I"', rstan.inc.path, '"')

RSTAN_LIBS <- paste0(' -L"', rstan.libs.path, '" -Wl,-rpath,"', rstan.libs.path, '" -lstan ')

# cat(PKG_CPPFLAGS_env, "\n")
# cat(RSTAN_LIBS, "\n") 

rstanplugin <- function() {
  Rcpp_plugin <- getPlugin("Rcpp") 
  list(includes = '', 
       body = function(x) x, 
       LinkingTo = c("Rcpp"),
       env = list(PKG_LIBS = paste(Rcpp_plugin$env$PKG_LIBS, RSTAN_LIBS), 
                  PKG_CPPFLAGS = paste(Rcpp_plugin$env$PKG_CPPFLAGS, PKG_CPPFLAGS_env)))  
} 

# registerPlugin("rstan", rstanplugin); 

inlineCxxPlugin <- function(...) {
  settings <- rstanplugin()  
  settings
}

