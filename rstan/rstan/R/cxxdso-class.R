setGeneric(name = "grab.cxxfun",
           def = function(object, ...) { standardGeneric("grab.cxxfun")})

setGeneric(name = "is.dso.loaded",
           def = function(object, ...) { standardGeneric("is.dso.loaded")})
setGeneric(name = "reload.dso", 
           def = function(object, ...) { standardGeneric("reload.dso")})

setMethod("show", "cxxdso", 
          function(object) {
            cat("S4 class cxxdso: dso.saved = ", object@dso.saved, 
                ", dso.filename = ", object@dso.filename, 
                ", size = ", obj.size.str(object.size(object@.MISC$dso.bin)), ".\n", sep = '')  
            cat("And dso.last.path = '", object@.MISC$dso.last.path, "'.\n", sep = '')
            cat("Created on: ", object@system, ".\n", sep = '')
            cat("Loaded now: ", is.dso.loaded(object), ".\n", sep = '')
            cat("The signatures is/are as follows: \n")
            print(object@sig); 
          })


setMethod('is.dso.loaded', signature(object = 'cxxdso'), 
          function(object) {
            f2 <- sub("\\.[^.]*$", "", basename(object@.MISC$dso.last.path)) 
            dlls <- getLoadedDLLs()
            f2 %in% names(dlls)
          }) 

setMethod('grab.cxxfun', signature(object = "cxxdso"), 
          function(object) { 
            if (!is.null.cxxfun(object@.MISC$cxxfun)) 
              return(object@.MISC$cxxfun)
            if (!object@dso.saved) 
              stop("the cxx fun is NULL now and this cxxdso is not saved")

            # If the file is still loaded  
            # from the help of function dyn.load 
            #   The function ‘dyn.unload’ unlinks the DLL.  Note that unloading a
            #   DLL and then re-loading a DLL of the same name may or may not
            #   work: on Solaris it uses the first version loaded.
            f <- sub("\\.[^.]*$", "", basename(object@dso.filename)) 
            f2 <- sub("\\.[^.]*$", "", basename(object@.MISC$dso.last.path)) 
            dlls <- getLoadedDLLs()
            if (f2 %in% names(dlls)) { # still loaded 
              DLL <- dlls[[f2]] 
              return(cxxfun.from.dll(object@sig, object@.MISC$cxxfun@code, DLL, check.dll = FALSE)) 
            }
            
            # not loaded  
            if (!identical(object@system, R.version$system)) 
              stop(paste("this cxxdso object was created on system '", object@system, "'", sep = ''))
            cx <- cxxfun.from.dso.bin(object) 
            assign('cxxfun', cx, object@.MISC) 
            return(cx) 
          }) 

setMethod("getDynLib", signature(x = "cxxdso"),
          function(x) { 
            fx <- grab.cxxfun(x) 
            env <- environment(fx@.Data)
            f <- get("f", env)
            dlls <- getLoadedDLLs()
            if (!f  %in% names(dlls)) 
              stop(paste('dso ', f, ' is not loaded', sep = ''))
            dlls[[f]]
          })
