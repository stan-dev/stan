
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
