/**
 * Define Rcpp Module to expose stan_fit's functions to R. 
 */ 
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::stan_fit<%model_name%_namespace::%model_name%,
               boost::random::ecuyer1988> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor<SEXP, SEXP>() 
    .method("call_sampler", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::call_sampler) 
    .method("get_first_p", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_first_p) 
    ;
}
