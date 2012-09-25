setGeneric(name = "grab_cxxfun",
           def = function(object, ...) { standardGeneric("grab_cxxfun")})

setGeneric(name = "is_dso_loaded",
           def = function(object, ...) { standardGeneric("is_dso_loaded")})
setGeneric(name = "reload_dso", 
           def = function(object, ...) { standardGeneric("reload_dso")})

setMethod("show", "cxxdso", 
          function(object) {
            cat("S4 class cxxdso: dso_saved = ", object@dso_saved, 
                ", dso_filename = ", object@dso_filename, 
                ", size = ", obj_size_str(object.size(object@.CXXDSOMISC$dso_bin)), ".\n", sep = '')  
            cat("And dso_last_path = '", object@.CXXDSOMISC$dso_last_path, "'.\n", sep = '')
            cat("Created on: ", object@system, " with ", object@cxxflags, ".\n", sep = '')
            cat("Loaded now: ", if (is_dso_loaded(object)) 'YES' else 'NO', ".\n", sep = '')
            cat("The signatures is/are as follows: \n")
            print(object@sig); 
          })

setMethod('is_dso_loaded', signature(object = 'cxxdso'), 
          function(object) {
            f2 <- sub("\\.[^.]*$", "", basename(object@.CXXDSOMISC$dso_last_path)) 
            dlls <- getLoadedDLLs()
            f2 %in% names(dlls)
          }) 

setMethod('grab_cxxfun', signature(object = "cxxdso"), 
          function(object) { 
            if (!is_null_cxxfun(object@.CXXDSOMISC$cxxfun)) 
              return(object@.CXXDSOMISC$cxxfun)
            if (!object@dso_saved) 
              stop("the cxx fun is NULL now and this cxxdso is not saved")

            # If the file is still loaded  
            # from the help of function dyn.load 
            #   The function ‘dyn.unload’ unlinks the DLL.  Note that unloading a
            #   DLL and then re-loading a DLL of the same name may or may not
            #   work: on Solaris it uses the first version loaded.
            f <- sub("\\.[^.]*$", "", basename(object@dso_filename)) 
            f2 <- sub("\\.[^.]*$", "", basename(object@.CXXDSOMISC$dso_last_path)) 
            dlls <- getLoadedDLLs()
            if (f2 %in% names(dlls)) { # still loaded 
              DLL <- dlls[[f2]] 
              fx <- cxxfun_from_dll(object@sig, object@.CXXDSOMISC$cxxfun@code, DLL, check_dll = FALSE) 
              assign('cxxfun', fx, envir = object@.CXXDSOMISC) 
              if (!is.null(object@modulename) && object@modulename != '') 
                assign("module", Module(object@modulename, getDynLib(fx)), envir = object@.CXXDSOMISC)
              return(fx) 
            }
            
            # not loaded  
            if (!identical(object@system, R.version$system)) 
              stop(paste("this cxxdso object was created on system '", object@system, "'", sep = ''))
            fx <- cxxfun_from_dso_bin(object) 
            assign('cxxfun', fx, envir = object@.CXXDSOMISC) 
            if (!is.null(object@modulename) && object@modulename != '') 
              assign("module", Module(object@modulename, getDynLib(fx)), envir = object@.CXXDSOMISC)
            return(fx) 
          }) 

setMethod("getDynLib", signature(x = "cxxdso"),
          function(x) { 
            fx <- grab_cxxfun(x) 
            env <- environment(fx@.Data)
            f <- get("f", env)
            dlls <- getLoadedDLLs()
            if (!f  %in% names(dlls)) 
              stop(paste('dso ', f, ' is not loaded', sep = ''))
            dlls[[f]]
          })
