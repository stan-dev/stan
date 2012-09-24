cxxfun_from_dll <- function(sig, code, DLL, check_dll = TRUE) { 
  # Create function objects from dll (most of the code are copied from
  # cxxfunction in package inline). 
  # 
  # Args:
  #  sig: a list of function signatures 
  #  DLL: object of class "DLLInfo"
  #  check_dll: check if the dll is loaded: When it is not 
  #    loaded, the function call might result in a segfault. 

  f <- DLL[['name']] 
  if (check_dll) {
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

cxxfun_from_dso_bin <- function(dso) {
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
  code <- dso@.CXXDSOMISC$cxxfun@code
  tfile <- tempfile() 
  f <- basename(tfile) 
  libLFile <- paste(tfile, ".", filename_ext(dso@.CXXDSOMISC$dso_last_path), sep = '') 
  # write the raw vector containing the dso file to temporary file
  writeBin(dso@.CXXDSOMISC$dso_bin, libLFile) 
  cleanup <- function(env) {
    if (f %in% names(getLoadedDLLs())) dyn.unload(libLFile)
      unlink(libLFile)
  }
  reg.finalizer(environment(), cleanup, onexit = TRUE)
  DLL <- dyn.load(libLFile) 
  assign('dso_last_path', libLFile, dso@.CXXDSOMISC) 
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


dso_path <- function(fx) {
  # find the path for the dynamic shared objects associated with 
  # the returned object from cxxfunction 
  # 
  # Args:
  #   fx: returned object from cxxfunction in package inline 
  dllinfo <- getDynLib(fx)
  dllinfo[['path']] 
} 

read_dso <- function(path) {
  n <- file.info(path)$size
  readBin(path, what = 'raw', n = n)
} 

cxxfunctionplus <- function(sig = character(), body = character(),
                            plugin = "default", includes = "",
                            settings = getPlugin(plugin), 
                            save_dso = FALSE, module_name = "MODULE", 
                            ..., verbose = FALSE) {
  fx <- cxxfunction(sig = sig, body = body, plugin = plugin, includes = includes, 
                    settings = settings, ..., verbose = verbose)
  dso_last_path <- dso_path(fx)
  dso_bin <- if (save_dso) read_dso(dso_last_path) else raw(0)
  dso_filename <- sub("\\.[^.]*$", "", basename(dso_last_path)) 
  if (!is.list(sig))  { 
    sig <- list(sig) 
    names(sig) <- dso_filename 
  } 
  dso <- new('cxxdso', sig = sig, dso_saved = save_dso, 
             dso_filename = dso_filename, 
             modulename = module_name, 
             system = R.version$system, 
             cxxflags = get_curr_cxxflags(), 
             .CXXDSOMISC = new.env()) 
  assign("cxxfun", fx, envir = dso@.CXXDSOMISC)
  assign("dso_last_path", dso_last_path, envir = dso@.CXXDSOMISC)
  assign("dso_bin", dso_bin, envir = dso@.CXXDSOMISC)
  if (!is.null(module_name) && module_name != '') 
    assign("module", Module(module_name, getDynLib(fx)), envir = dso@.CXXDSOMISC)
  return(dso)
} 

# write_dso 
# writeBin(dso, '/tmp/Rtmpdb9w5A/aa.so')
