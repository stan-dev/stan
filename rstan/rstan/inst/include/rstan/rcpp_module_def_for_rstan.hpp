/**
 * Define Rcpp Module to expose nuts_r_ui's functions to R. 
 */ 
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::nuts_r_ui<%model_name%_namespace::%model_name%,
               boost::random::ecuyer1988> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor<SEXP, SEXP>() 
    .method("call_nuts", 
            &rstan::nuts_r_ui<%model_name%_namespace::%model_name%,
                              boost::random::ecuyer1988>::call_nuts) 
    .method("get_first_p", 
            &rstan::nuts_r_ui<%model_name%_namespace::%model_name%,
                              boost::random::ecuyer1988>::get_first_p) 
    ;
}
