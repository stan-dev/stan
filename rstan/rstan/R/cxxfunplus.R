cxxfun.from.dll <- function(sig, code, DLL, check.dll = TRUE) { 
  # Create function objects from dll (most of the code are copied from
  # cxxfunction in package inline). 
  # 
  # Args:
  #  sig: a list of function signatures 
  #  DLL: object of class "DLLInfo"
  #  check.dll: check if the dll is loaded: When it is not 
  #    loaded, the function call might result in a segfault. 

  f <- DLL[['name']] 
  if (check.dll) {
    dlls <- getLoadedDLLs()
    if (!f %in% names(dlls)) 
      stop(paste("dso ", DLL[['path']], " is not loaded", sep = ''))
  } 

  res <- vector("list", length(sig))
  names(res) <- names(sig)
  res <- new("CFuncList", res)
 
  for(i in seq_along(sig)) {
    res[[i]] <- new("CFunc", code = code)
    fn <- function(arg) { NULL }

    ## Modify the function formals to give the right argument list
    args <- formals(fn)[rep(1, length(sig[[i]]))]
    names(args) <- names(sig[[i]])
    formals(fn) <- args

    ## create .Call function call that will be added to 'fn'
    body <- quote(.Call(EXTERNALNAME, ARG))[c(1:2, rep(3, length(sig[[i]])))]
    for (j in seq(along = sig[[i]])) body[[j + 2]] <- as.name(names(sig[[i]])[j])

    body[[1L]] <- .Call
    body[[2L]] <- getNativeSymbolInfo(names(sig)[[i]], DLL)$address
    ## update the body of 'fn'
    body(fn) <- body
    ## set fn as THE function in CFunc of res[[i]]
    res[[i]]@.Data <- fn
  }
  ## clear the environment
  rm(j)
  convention <- ".Call"
  if (identical(length(sig), 1L)) res[[1L]] else res
} 

cxxfun.from.dso.bin <- function(dso) {
  # Create function objects from dll (most of the code are copied from
  # cxxfunction in package inline). 
  # 
  # Args:
  #  dso: object of class cxxdso 
  # 
  # Note: we are assuming that the dso is not loaded so
  #   we create the dso file from the raw vector 
  #   and then loaded the dso. . 

  sig <- dso@sig 
  code <- dso@.MISC$cxxfun@code
  tfile <- tempfile() 
  f <- basename(tfile) 
  libLFile <- paste(tfile, ".", file_ext(dso@.MISC$dso.last.path), sep = '') 
  # write the raw vector containing the dso file to temporary file
  writeBin(dso@.MISC$dso.bin, libLFile) 
  cleanup <- function(env) {
    if (f %in% names(getLoadedDLLs())) dyn.unload(libLFile)
      unlink(libLFile)
  }
  reg.finalizer(environment(), cleanup, onexit = TRUE)
  DLL <- dyn.load(libLFile) 
  assign('dso.last.path', libLFile, dso@.MISC) 
  res <- vector("list", length(sig))
  names(res) <- names(sig)
  res <- new("CFuncList", res)
  for(i in seq_along(sig)) {
    res[[i]] <- new("CFunc", code = code) 
    fn <- function(arg) { NULL }

    ## Modify the function formals to give the right argument list
    args <- formals(fn)[rep(1, length(sig[[i]]))]
    names(args) <- names(sig[[i]])
    formals(fn) <- args

    ## create .Call function call that will be added to 'fn'
    body <- quote(.Call(EXTERNALNAME, ARG))[c(1:2, rep(3, length(sig[[i]])))]
    for (j in seq(along = sig[[i]])) body[[j + 2]] <- as.name(names(sig[[i]])[j])

    body[[1L]] <- .Call
    body[[2L]] <- getNativeSymbolInfo(names(sig)[[i]], DLL)$address
    ## update the body of 'fn'
    body(fn) <- body
    ## set fn as THE function in CFunc of res[[i]]
    res[[i]]@.Data <- fn
  }
  ## clear the environment
  rm(j)
  rm(tfile) 
  convention <- ".Call"
  if (identical(length(sig), 1L)) res[[1L]] else res
} 


dso.path <- function(fx) {
  # find the path for the dynamic shared objects associated with 
  # the returned object from cxxfunction 
  # 
  # Args:
  #   fx: returned object from cxxfunction in package inline 
  dllinfo <- getDynLib(fx)
  dllinfo[['path']] 
} 

read.dso <- function(path) {
  n <- file.info(path)$size
  readBin(path, what = 'raw', n = n)
} 

cxxfunctionplus <- function(sig = character(), body = character(),
                            plugin = "default", includes = "",
                            settings = getPlugin(plugin), 
                            save.dso = FALSE, ..., verbose = FALSE) {
  fx <- cxxfunction(sig = sig, body = body, plugin = plugin, includes = includes, 
                    settings = settings, ..., verbose = verbose)
  dso.last.path <- dso.path(fx)
  dso.bin <- if (save.dso) read.dso(dso.last.path) else raw(0)
  dso.filename <- sub("\\.[^.]*$", "", basename(dso.last.path)) 
  if (!is.list(sig))  { 
    sig <- list(sig) 
    names(sig) <- dso.filename 
  } 
  dso <- new('cxxdso', sig = sig, dso.saved = save.dso, 
             dso.filename = dso.filename, 
             system = R.version$system, 
             .MISC = new.env()) 
  assign("cxxfun", fx, envir = dso@.MISC)
  assign("dso.last.path", dso.last.path, envir = dso@.MISC)
  assign("dso.bin", dso.bin, envir = dso@.MISC)
  return(dso)
} 

# write.dso 
# writeBin(dso, '/tmp/Rtmpdb9w5A/aa.so')
