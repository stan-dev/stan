# to copy rstan.so to libs/R_ARCH 
# and copy libstan.a to libstan/R_ARCH

# REF: http://cran.r-project.org/doc/manuals/R-exts.html#Package-subdirectories


WINDOWS <- .Platform$OS.type == "windows"

## to copy rstan.so (code is copied from tools/install.R) 
files <- Sys.glob(paste("*", SHLIB_EXT, sep = ''))
if (length(files)) { 
  libarch <- if (nzchar(R_ARCH)) paste('libs', R_ARCH, sep = '') else 'libs'
  dest <- file.path(R_PACKAGE_DIR, libarch)
  message('installing rstan libs to ', dest)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  file.copy(files, dest, overwrite = TRUE)
  ## not clear if this is still necessary, but sh version did so
  if (!WINDOWS) Sys.chmod(file.path(dest, files), "755")

  ## for the mac, the dSYM can be created if --dsym option  
  ## is specified. 
  dsym <- nzchar(Sys.getenv("PKG_MAKE_DSYM"))
  args <- commandArgs(TRUE)
  args <- strsplit(args, 'nextArg', fixed = TRUE)[[1L]][-1]
  if ('--dsym' %in% args)  dsym <- TRUE

  if (dsym && length(grep("^darwin", R.version$os))) {
    message('generating debug symbols (dSYM)')
    dylib <- Sys.glob(paste(dest, "/*", SHLIB_EXT, sep = ''))
    for (file in dylib) system(paste("dsymutil ", file, sep = ''))
  }
 
  if (file_test("-f", "symbols.rds")) file.copy("symbols.rds", dest)
} 


## to copy libstan.a 
STATICLIB_EXT <- 'a' 
files <- Sys.glob(paste("*", STATICLIB_EXT, sep = ''))
if (length(files)) { 
  libstanarch <- if (nzchar(R_ARCH)) paste('libstan', R_ARCH, sep = '') else 'libstan'
  dest <- file.path(R_PACKAGE_DIR, libstanarch)
  message('installing libstan.a to ', dest)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  file.copy(files, dest, overwrite = TRUE)
} 

