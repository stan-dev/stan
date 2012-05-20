/**
 * This is not a C++ code, but a template of C++ code 
 * for defining Rcpp modules. 
 *
 * The module name is the `model_name', which 
 * should be replaced to the real model_name
 * in the end. That is, %model_name% here would be
 * changed say to anon_model.  
 *
 */ 
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::nuts_r_ui<%model_name%_namespace::%model_name%> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor() 
    .method("call_nuts", 
            &rstan::nuts_r_ui<%model_name%_namespace::%model_name%>::call_nuts) 
    ;
}
