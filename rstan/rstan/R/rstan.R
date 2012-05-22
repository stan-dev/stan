
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
      .modelData = list(nuts = modelnuts))  

} 

## @stan.m: what stan.m should be?? of stanmodel class??
## @param init.t: '0', 'user', any other values indicate `random' 
stan.samples <- function(stan.m, 
                         data, 
                         n.chains = 1, # chain.id ?? 
                         n.iter = 2000, 
                         seed, 
                         thin = 1, 
                         init.t = 'random', 
                         init.v = NULL, ...,
                         verbose = FALSE) {

   # check data and preprocess 
   data <- data.preprocess(data) 
   if (missing(init.t) || is.na(init.t)) {
     init_t <- "random"
   } else if (init.t == 0 || init.t == '0')  { 
     init_t <- "0"; 
   } else if (init.t != 'user') {
     init_t <= 'random' 
   } 

   args <- list(iter = n.iter, init = init.t, init_lst = init.v) 
   if (!missing(seed)) 
     args$seed = seed  
                
   # check inits and construct inits 
   # call stan.m's nuts_command 

   # check if stan.m is valid? (How?)
   stan.m$call_nuts(data, args) 
}  
