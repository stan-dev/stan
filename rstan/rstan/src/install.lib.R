# to copy rstan.so to libs/R_ARCH 
# and copy libstan.a to libstan/R_ARCH

# REF: http://cran.r-project.org/doc/manuals/R-exts.html#Package-subdirectories

files <- Sys.glob(paste("*", SHLIB_EXT, sep = ''))
libarch <- if (nzchar(R_ARCH)) paste('libs', R_ARCH, sep = '') else 'libs'
dest <- file.path(R_PACKAGE_DIR, libarch)
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
file.copy(files, dest, overwrite = TRUE)

STATICLIB_EXT <- 'a' 
libstanarch <- if (nzchar(R_ARCH)) paste('libstan', R_ARCH, sep = '') else 'libstan'
dest <- file.path(R_PACKAGE_DIR, libarch)
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
file.copy(files, dest, overwrite = TRUE)

