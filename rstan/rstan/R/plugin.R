# Part of the rstan package for an R interface to Stan 
# Copyright (C) 2012 Columbia University
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


##
## Define rstan plugin for inline package.
## (original name: inline.R)
##

rstan.inc.path_fun <- function() { 
  system.file('include', package = 'rstan')
} 

rstan.libs.path_fun <- function() {
  if (nzchar(.Platform$r_arch)) {
    return(system.file('libstan', .Platform$r_arch, package = 'rstan'))
  }
  system.file('libstan', package = 'rstan')
}

# Using RcppEigen
eigen.path_fun <- function() {
  system.file('include', package = 'RcppEigen')
} 

# If included in RStan
# eigen.path_fun() <- paste0(rstan.inc.path_fun(), '/stanlib/eigen_3.1.0')

static.linking <- function() {
  # return(Rcpp:::staticLinking());
  ## not following Rcpp's link, we only have either dynamic version or static
  ## version because the libraries are big.
  ## (In Rcpp, both versions are compiled.)
  # return(.Platform$OS.type == 'windows')
  # For the time being, use static linking for libstan on all platforms. 
  TRUE
}

PKG_CPPFLAGS_env_fun <- function() {
   paste(' -I"', file.path(rstan.inc.path_fun(), '/stansrc" '),
         ' -I"', file.path(eigen.path_fun(), '" '),
         ' -I"', file.path(eigen.path_fun(), '/unsupported" '),
         ' -I"', rstan.options("boost.lib"), '"',
         ' -I"', rstan.inc.path_fun(), '"', sep = '')
}

RSTAN_LIBS_fun <- function() {
  static <- static.linking() 
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
