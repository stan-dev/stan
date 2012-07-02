rstan.inc.path <- system.file('include', package = 'rstan')
PKG_CPPFLAGS_env <- paste0(' -I"', paste0(rstan.inc.path, '/stansrc" '), 
                           ' -I"', paste0(rstan.inc.path, '/stansrc" '), 
                           ' -I"', rstan.inc.path, '"')
# print(PKG_CPPFLAGS_env)

rstanplugin <- function() {
  Rcpp_plugin <- getPlugin("Rcpp") 
  list(includes = '', 
       body = function(x) x, 
       LinkingTo = c("Rcpp"),
       env = list(PKG_LIBS = Rcpp_plugin$env$PKG_LIBS, 
                  PKG_CPPFLAGS = paste(Rcpp_plugin$env$PKG_CPPFLAGS, PKG_CPPFLAGS_env)))  
} 

# registerPlugin("rstan", rstanplugin); 

inlineCxxPlugin <- function(...) {
  settings <- rstanplugin()  
  settings
}

