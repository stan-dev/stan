
stan.model <- function(file, verbose = FALSE, 
                       model.name = "anon_model", 
                       model.code = '') {

  model.code <- get.model.code(file, model.code)  
  r <- stanc(model.code, model.name); 
  inc <- paste("#include <rstan/nuts_r_ui.hpp>\n", 
               r$cppcode, 
               get_Rcpp_module_def_code(model.name), 
               sep = ''); 
  
  fx <- cxxfunction(signature(), body = '  return R_NilValue;', 
                    include = inc, plugin = "rstan", verbose = verbose) 
               

  mod <- Module(model.name, getDynLib(fx)) 
  modelnuts <- do.call("$", list(mod, model.name))
  new("stanmodel", model.name = model.name, 
      .modelData = list(nuts = modelnuts))  

} 

stan.sample <- function(stan.m, 
                        n.chains = 1, # chain.id ?? 
                        n.iter, 
                        seed, 
                        data, init, 
                        verbose = FALSE) {

   # check data 
   # check inits and construct inits 
   # call stan.m's nuts_command 
}  
