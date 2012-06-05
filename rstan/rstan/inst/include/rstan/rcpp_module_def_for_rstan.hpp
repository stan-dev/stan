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

    .method("get_chain_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_samples) 

    .method("get_kept_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_kept_samples) 

    .method("get_chain_kept_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_kept_samples) 

    .method("param_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::param_names) 

    .method("warmup", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::warmup) 

    .method("num_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::num_samples) 

    .method("get_summary", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_summary) 

    .method("get_summary_item_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_summary_item_names) 

    ;
}
