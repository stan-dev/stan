##
## Define rstan plugin for inline package.
## (original name: inline.R)
##

rstan_inc_path_fun <- function() { 
  system.file('include', package = 'rstan')
} 

rstan_libs_path_fun <- function() {
  if (nzchar(.Platform$r_arch)) {
    return(system.file('libstan', .Platform$r_arch, package = 'rstan'))
  }
  system.file('libstan', package = 'rstan')
}

# Using RcppEigen
eigen_path_fun <- function() {
  rstan_options("eigen_lib")
} 

# If included in RStan
# eigen_path_fun() <- paste0(rstan_inc_path_fun(), '/stanlib/eigen_3.1.0')

static_linking <- function() {
  # return(Rcpp:::staticLinking());
  ## not following Rcpp's link, we only have either dynamic version or static
  ## version because the libraries are big.
  ## (In Rcpp, both versions are compiled.)
  # return(.Platform$OS.type == 'windows')
  # For the time being, use static linking for libstan on all platforms. 
  TRUE
}

PKG_CPPFLAGS_env_fun <- function() {
   paste(' -I"', file.path(rstan_inc_path_fun(), '/stansrc" '),
         ' -I"', file.path(eigen_path_fun(), '" '),
         ' -I"', file.path(eigen_path_fun(), '/unsupported" '),
         ' -I"', rstan_options("boost_lib"), '"',
         ' -I"', rstan_inc_path_fun(), '"', sep = '')
}

legitimate_space_in_path <- function(path) {
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
  static <- static_linking() 
  rstan.libs.path <- rstan_libs_path_fun()

  # It seems that adding quotes to the path does not work well 
  # in the case there is space in the path name 
  if (grepl('[^\\\\]\\s', rstan.libs.path, perl = TRUE))
    rstan.libs.path <- legitimate_space_in_path(rstan.libs.path)

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
  rcpp_pkg_path2 <- legitimate_space_in_path(rcpp_pkg_path) 
 
  # In case  we have space (typicall on windows though not necessarily)
  # in the file path of Rcpp's library. 
  
  # If rcpp_PKG_LIBS contains space without preceding '\\', add `\\'; 
  # otherwise keept it intact
  if (grepl('[^\\\\]\\s', rcpp_pkg_libs, perl = TRUE))
    rcpp_pkg_libs <- gsub(rcpp_pkg_path, rcpp_pkg_path2, rcpp_pkg_libs, fixed = TRUE) 

  list(includes = '',
       body = function(x) x,
       LinkingTo = c("Rcpp"),
       ## FIXME see if we can use LinkingTo for RcppEigen's header files
       env = list(PKG_LIBS = paste(rcpp_pkg_libs, RSTAN_LIBS_fun()),
                  PKG_CPPFLAGS = paste(Rcpp_plugin$env$PKG_CPPFLAGS, PKG_CPPFLAGS_env_fun())))
}


# inlineCxxPlugin would automatically get registered in inline's plugin list.
# Note that everytime rstan plugin is used, inlineCxxPlugin
# gets called so we can change some settings on the fly
# for example now by setting rstan_options(boost_lib=xxx)
inlineCxxPlugin <- function(...) {
  settings <- rstanplugin()
  settings
}


## deal with the optimization level for c++ compilation 
get_curr_cxxflags <- function() { 
  # get current CXXFLAGS used for in R CMD SHLIB, which controls
  # how the generated C++ code for stan model are compiled 
  # at which optimization level. 
  # The order or files to look for are following the code 
  # of function .SHLIB in packages tools/R/install.R 
  
  # Return: 
  #   the whole line that defines CXXFLAGS 

  WINDOWS <- .Platform$OS.type == "windows"
  rarch <- Sys.getenv("R_ARCH") # unix only
  if (WINDOWS && nzchar(.Platform$r_arch))
    rarch <- paste0("/", .Platform$r_arch)
  site <- file.path(R.home("etc"), rarch, "Makevars.site")

  makefiles <-
    c(file.path(R.home("etc"), rarch, "Makeconf"),
      if (file.exists(site)) site,
      file.path(R.home("share"), "make", if (WINDOWS) "winshlib.mk" else "shlib.mk"))

  if (WINDOWS) { 
    if (rarch == "/x64" && 
        file.exists(f <- path.expand("~/.R/Makevars.win64"))) 
      makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars.win"))) 
      makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars"))) 
      makefiles <- c(makefiles, f) 
  } else { 
    if (file.exists(f <- path.expand(paste("~/.R/Makevars", 
                                            Sys.getenv("R_PLATFORM"), 
                                            sep = "-")))) 
       makefiles <- c(makefiles, f) 
    else if (file.exists(f <- path.expand("~/.R/Makevars"))) 
       makefiles <- c(makefiles, f) 
  }

  makefile_txt <- do.call(c, lapply(makefiles, function(f) readLines(f, warn = FALSE))) 

  lineno <- -1 
  for (i in 1:length(makefile_txt)) { 
    if (grepl("^\\s*CXXFLAGS\\s*=", makefile_txt[i])) lineno <- i 
  } 

  if (-1 != lineno) return(makefile_txt[lineno]) 
  return("CXXFLAGS = ") 
} 

last_makefile <- function() {
  # Return the last makefile for 'R CMD SHLIB'. 
  # In essential, R CMD SHLIB uses GNU make.  
  # R CMD SHLIB uses a series of files as the whole makefile.  
  # This function return the last one, where we can set flags 
  # to overwrite what is set before. 
  # 
  WINDOWS <- .Platform$OS.type == "windows"
  rarch <- Sys.getenv("R_ARCH") # unix only
  if (WINDOWS && nzchar(.Platform$r_arch))
    rarch <- paste0("/", .Platform$r_arch)

  if (WINDOWS) { 
    if (rarch == "/x64") return(path.expand("~/.R/Makevars.win64")) 
    else return(path.expand("~/.R/Makevars.win"))
  }
  path.expand(paste("~/.R/Makevars", Sys.getenv("R_PLATFORM"), sep = "-"))
} 

set_cxxflags <- function(cxxflags) {
  # Set user-defined CXXFLAGS for R CMD SHLIB by 
  # set CXXFLAGS in the last makefile obtained by last_makefile 
  # Args: 
  #   cxxflags: the whole CXXFLAGS such as 'CXXFLAGS = -O3' 
  # 
  # Return: 
  #   TRUE if it is successful, otherwise, the function
  #   stops and reports an error. 
  # 
  lmf <- last_makefile() 
  homedotR <- dirname(lmf) 
  if (!file.exists(homedotR)) {
    if (!dir.create(homedotR, showWarnings = FALSE, recursive = TRUE))
      stop(paste("failed to create directory ", homedotR, sep = '')) 
  } 

  if (!file.exists(lmf)) {
    if (file.access(homedotR, mode = 2) < 0 || 
        file.access(homedotR, mode = 1) < 0 ) 
      stop(paste("directory ", homedotR, " is not writable", sep = '')) 
    cat(paste("# created by rstan at ", date(), sep = ''), 
        cxxflags, sep = '\n', file = lmf)
    return(invisible(TRUE)) 
  } 

  if (file.access(lmf, mode = 2) < 0) 
    stop(paste(lmf, " is not writable", sep = '')) 

  lmf_txt <- readLines(lmf, warn = FALSE) 
  if (length(lmf_txt) < 1) {  # empty file 
    cat(cxxflags, file = lmf, append = TRUE) 
    return(invisible(TRUE))  
  } 
  found <- FALSE 
  for (i in length(lmf_txt):1) {
    if (grepl("^\\s*CXXFLAGS\\s*=", lmf_txt[i])) {
      lmf_txt[i] <- cxxflags
      found <- TRUE 
      break; 
    } 
  } 
  if (found) cat(paste(lmf_txt, collapse = '\n'), file = lmf) 
  else cat(cxxflags, file = lmf, append = TRUE) 
  invisible(TRUE) 
} 


rm_last_makefile <- function() {
  # remove the file given by last_makefile()
  # return: 
  #  ‘0’ for success, ‘1’ for failure, invisibly. 
  #  '-1' file not exists 
  lmf <- last_makefile() 
  if (file.exists(lmf)) 
    return(unlink(lmf, force = TRUE)) 
  return(-1) 
} 


get_cxxo_level <- function(str) {
  # obtain the optimization level from a string like -O3 
  # Return: 
  #   the optimization level after -O as a character 
  # 
  p <- regexpr('-O.', str, perl = TRUE) + 2 
  if (p == 1) return("")  # not found 
  substr(str, p, p)
} 

set_cppo <- function(level = c("3", "2", "1", "0")) {
  # set the optimization level for compiling C++ code using R CMD SHLIB
  # Args:
  #   level: one of "0", "1", "2", "3"
  level = match.arg(level) 
  curr_cxxflags <- get_curr_cxxflags() 
  if (is.numeric(level)) 
    level <- as.character(level) 
  if (!level %in% c('0', '1', '2', '3', 's')) 
    stop(paste(level, " might not be a legal optimization level for C++ compilation", sep = '')) 
  old_olevel <- get_cxxo_level(curr_cxxflags) 
  if (old_olevel == level)  return(curr_cxxflags) 
  newcxxflags <- if (old_olevel == '')  { 
                   paste(curr_cxxflags, " -O", level, sep = '') 
                 } else {
                   sub(paste("-O", old_olevel, sep = ''), 
                       paste("-O", level, sep = ''), curr_cxxflags, fixed = TRUE) 
                 } 
  set_cxxflags(newcxxflags) 
  newcxxflags 
} 

get_cppo <- function() {
  # get the optimization level as a character 
  curr_cxxflags <- get_curr_cxxflags() 
  get_cxxo_level(curr_cxxflags)  
} 
