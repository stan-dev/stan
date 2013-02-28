/**
 * Define Rcpp Module to expose stan_fit's functions to R. 
 */ 
RCPP_MODULE(stan_fit4%model_name%_mod){
  Rcpp::class_<rstan::stan_fit<%model_name%_namespace::%model_name%, 
               boost::random::ecuyer1988> >("stan_fit4%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor<SEXP, SEXP>() 
    // .constructor<SEXP, SEXP>() 
    .method("call_sampler", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::call_sampler) 
    .method("param_names", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_names) 
    .method("param_names_oi", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_names_oi) 
    .method("param_fnames_oi", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_fnames_oi) 
    .method("param_dims", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_dims) 
    .method("param_dims_oi", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_dims_oi) 
    .method("update_param_oi", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::update_param_oi) 
    .method("param_oi_tidx",
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_oi_tidx)
    .method("grad_log_prob", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::grad_log_prob) 
    .method("log_prob", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::log_prob) 
    .method("transform_pars", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::transform_pars) 
    .method("constrain_pars", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::constrain_pars) 
    .method("num_pars_unconstrained", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::num_pars_unconstrained)
    ;
} 
