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

    .method("get_param_dimss", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_param_dimss) 

    .method("get_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_samples) 

    .method("get_chain_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_samples) 

    .method("param_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::param_names) 

    .method("param_flat_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::param_flat_names) 

    .method("warmup", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::warmup) 

    .method("num_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::num_samples) 

    .method("num_kept_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::num_kept_samples) 

    .method("num_chain_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::num_chain_samples) 

    .method("num_chain_kept_samples", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::num_chain_kept_samples) 

    .method("get_mean_and_sd", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_mean_and_sd) 

    .method("get_chain_mean_and_sd", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_mean_and_sd) 

    .method("get_quantiles", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_quantiles)   

    .method("get_chain_quantiles", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_quantiles)   

    .method("get_ess", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_ess) 

    .method("get_split_rhat", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_split_rhat) 
    
    .method("get_kept_samples_permuted", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_kept_samples_permuted) 

    .method("get_chain_stan_args", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_chain_stan_args) 

    .method("get_stan_args", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%,
                             boost::random::ecuyer1988>::get_stan_args) 

    ;
}
