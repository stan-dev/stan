
setClass(Class = "cxxdso",
         representation = representation(
           sig = "list", # A list of function signature that would be returned by cxxfuncion 
           dso_saved = "logical", # flag for if the dso is saved or not
           # dso_last_path = 'character', # where the dso is saved last time 
           dso_filename = "character", # the dso file name when it is created the first time
           modulename = 'character', # in rstan, we always compile the c++ code with an Rcpp module
           system = "character", # what is the OS (R.version$system)?  
           cxxflags = "character", # the CXXFLAGS used to compile the DSO 
           .CXXDSOMISC = "environment" # an envir to save 
                                 #  1. the returned object by cxxfuncion using name cxxfun 
                                 #  2. the file path used last time using name dso_last_path 
                                 #  3. The binary dso with name dso_bin, which is a raw vector.  
                                 #     We put it here since the environment is not copied 
                                 #     when assigned to another 
                                 #     http://cran.r-project.org/doc/manuals/R-lang.html#Environment-objects
         ),
         validity = function(object) {
           length(object@sig) > 0 && identical(object@system, R.version$system)
         })

setClass(Class = "stanmodel",
         representation = representation(
           model_name = "character",
           model_code = "character",
           model_cpp = "list", 
             # model_cppname (used to define Rcpp module)  & 
             # model_cppcode (just the C++ code for the model) 
           dso = 'cxxdso'), 
         validity = function(object) {
           return(TRUE)
         })


setClass(Class = "stanfit",
         representation = representation(
           model_name = "character", 
           model_pars = "character", 
           par_dims = "list", 
           mode = "integer", # 0: samples; 1: test_grad (no samples); 2: other error (no samples) 
           sim = "list", 
           inits = "list", 
           stan_args = "list", 
           stanmodel = "stanmodel", # the instance of S4 class stanmodel 
           date = "character", # the date samples were drawn 
           .MISC = "environment"
         ),  
         validity = function(object) {
           return(TRUE) 
         })

# list all methods for stanfit 
# showMethods(class = 'stanfit', where = getNamespace("rstan"))
