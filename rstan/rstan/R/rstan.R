
## 
## @param stan.home TBD 
## 
stan.model <- function(file, verbose = FALSE, 
                       model.name = "anon_model", 
                       model.code = '', stan.home) {

  model.code <- get.model.code(file, model.code)  
  r <- stanc(model.code, model.name); 
  inc <- paste("#include <rstan/nuts_r_ui.hpp>\n", 
               r$cppcode, 
               get_Rcpp_module_def_code(model.name), 
               sep = ''); 
  
  fx <- cxxfunction(signature(), body = '  return R_NilValue;', 
                    includes = inc, plugin = "rstan", verbose = verbose) 
               

  mod <- Module(model.name, getDynLib(fx)) 
  # modelnuts <- do.call("$", list(mod, model.name))
  modelnuts <- eval(call("$", mod, model.name))
  new("stanmodel", model.name = model.name, 
      .modelmod = list(nuts = modelnuts))  

} 

## @stan.m: what stan.m should be?? of stanmodel class??
## @param init.t: '0', 'user', any other values indicate `random' 
stan.samples <- function(stan.m, 
                         data, 
                         n.chains = 1L, # chain.id ?? 
                         n.iter = 2000L, 
                         thin = 1L, 
                         init.t = 'random', 
                         init.v = NULL, 
                         seed, 
                         sample_file, ...,
                         verbose = FALSE) {

   # check data and preprocess 
   data <- data.preprocess(data) 

   # assemble init and init_lst 
   # init: how to set up the initial values (0, user, random)
   # init_lst: user specified initial values list  
   if (init.t == 0) 
     init.t <- "0"; 

   if (!init.t %in% c("0", "user", "random")) 
     init.t <- "random"; 
    
   args <- list(init = init.t, iter = n.iter, thin = thin) 

   if (init.t == 'user' && is.list(init.v)) 
     args$init_lst <- init.v 

   if (!missing(seed)) 
     args$seed <- seed  

   if (!missing(sample_file)) 
     args$sample_file <- sample_file  
                
   # check inits and construct inits 
   # call stan.m's nuts_command 

   # check if stan.m is valid? (How?)
   nuts <- new(stan.m@.modelmod$nuts) 
   invisible(nuts$call_nuts(data, args)) 
}  
