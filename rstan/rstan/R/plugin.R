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

legitimate.space.in.path <- function(path) {
  # Add preceding '\\' to spaces on non-windows (this should happen rarely,
  # and not sure it will work)
  # For windows, use the short path name (8.3 format) 
  # 
  WINDOWS <- .Platform$OS.type == "windows"
  if (WINDOWS) { 
    path2 <- utils::shortPathName(path) 
    # it is weird that the '\\' in the path name will be gone
    # when passed to cxxfunction, so chagne it to '/' 
    return(gsub('\\\\', '/', path2, perl=TRUE))
  }
  gsub("([^\\\\])(\\s+)", '\\1\\\\\\2', path, perl = TRUE)
} 

RSTAN_LIBS_fun <- function() {
  static <- static.linking() 
  rstan.libs.path <- rstan.libs.path_fun()

  # It seems that adding quotes to the path does not work well 
  # in the case there is space in the path name 
  if (grepl('[^\\\\]\\s', rstan.libs.path, perl = TRUE))
    rstan.libs.path <- legitimate.space.in.path(rstan.libs.path)

  if (static) {
    paste(' "', rstan.libs.path, '/libstan.a', '"', sep = '')
  } else {
    paste(' -L"', rstan.libs.path, '" -Wl,-rpath,"', rstan.libs.path, '" -lstan ', sep = '')
  }
}

# cat(PKG_CPPFLAGS_env, "\n")
# cat(RSTAN_LIBS_fun(), "\n")

rstanplugin <- function() {
  Rcpp_plugin <- getPlugin("Rcpp")
  rcpp_pkg_libs <- Rcpp_plugin$env$PKG_LIBS
  rcpp_pkg_path <- system.file(package = 'Rcpp')
  rcpp_pkg_path2 <- legitimate.space.in.path(rcpp_pkg_path) 
 
  # In case  we have space (typicall on windows though not necessarily)
  # in the file path of Rcpp's library. 
  
  # If rcpp_PKG_LIBS contains space without preceding '\\', add `\\'; 
  # otherwise keept it intact
  if (grepl('[^\\\\]\\s', rcpp_pkg_libs, perl = TRUE))
    rcpp_pkg_libs <- gsub(rcpp_pkg_path, rcpp_pkg_path2, rcpp_pkg_libs, fixed = TRUE) 

  list(includes = '',
       body = function(x) x,
       LinkingTo = c("Rcpp"),
       ## FIXME see if we can use LinkingTo for RcppEighen's header files
       env = list(PKG_LIBS = paste(rcpp_pkg_libs, RSTAN_LIBS_fun()),
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
