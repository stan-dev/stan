##
## Define rstan plugin for inline package.
##

rstan.inc.path  <- system.file('include', package = 'rstan')

rstan.libs.path_fun <- function() {
  if (nzchar(.Platform$r_arch)) {
    return(system.file('libstan', .Platform$r_arch, package = 'rstan'))
  }
  system.file('libstan', package = 'rstan')
}

# Using RcppEigen
eigen.path <- system.file('include', package = 'RcppEigen')

# If included in RStan
# eigen.path <- paste0(rstan.inc.path, '/stanlib/eigen_3.1.0')


PKG_CPPFLAGS_env_fun <- function() {
   paste(' -I"', file.path(rstan.inc.path, '/stansrc" '),
         ' -I"', file.path(eigen.path, '" '),
         ' -I"', file.path(eigen.path, '/unsupported" '),
         ' -I"', rstan.options("boost.lib"), '"',
         ' -I"', rstan.inc.path, '"', sep = '')
}

RSTAN_LIBS_fun <- function() {
  static <- Rcpp:::staticLinking()
  ## currently following Rcpp's link mechanism, which
  ## is not necessarily good in our case.
  rstan.libs.path <- rstan.libs.path_fun()
  if (static) {
    paste('"', rstan.libs.path, '/libstan.a', '"', sep = '')
  } else {
    paste(' -L"', rstan.libs.path, '" -Wl,-rpath,"', rstan.libs.path, '" -lstan ', sep = '')
  }
}

# cat(PKG_CPPFLAGS_env, "\n")
# cat(RSTAN_LIBS_fun(), "\n")

rstanplugin <- function() {
  Rcpp_plugin <- getPlugin("Rcpp")
  list(includes = '',
       body = function(x) x,
       LinkingTo = c("Rcpp"),
	   ## FIXME see if we can use LinkingTo for RcppEighen's header files
       env = list(PKG_LIBS = paste(Rcpp_plugin$env$PKG_LIBS, RSTAN_LIBS_fun()),
                  PKG_CPPFLAGS = paste(Rcpp_plugin$env$PKG_CPPFLAGS, PKG_CPPFLAGS_env_fun())))
}


# inlineCxxPlugin would automatically get registered in inline's plugin list.
# Note that everytime rstan plugin is used, inlineCxxPlugin
# gets called so we can change some settings on the fly
# for example now by setting rstan.options(boost.lib=xxx)
inlineCxxPlugin <- function(...) {
  settings <- rstanplugin()
  settings
}
