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
    .method("get_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_samples) 

    .method("param_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::param_names) 

    ;
}
