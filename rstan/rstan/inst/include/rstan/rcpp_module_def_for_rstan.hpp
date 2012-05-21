/**
 * Define Rcpp Module to expose nuts_r_ui's functions to R. 
 */ 
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::nuts_r_ui<%model_name%_namespace::%model_name%> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor() 
    .method("call_nuts", 
            &rstan::nuts_r_ui<%model_name%_namespace::%model_name%>::call_nuts) 
    ;
}
