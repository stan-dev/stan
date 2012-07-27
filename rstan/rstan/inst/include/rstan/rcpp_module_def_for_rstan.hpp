/**
 * Define Rcpp Module to expose stan_fit's functions to R. 
 */ 
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::stan_fit<%model_name%_namespace::%model_name%, 
               boost::random::ecuyer1988> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor<SEXP>() 
    .constructor<SEXP, SEXP>() 
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
    .method("permutation", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::permutation) 
    .method("update_param_oi", 
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::update_param_oi) 
    .method("param_oi_tidx",
            &rstan::stan_fit<%model_name%_namespace::%model_name%, boost::random::ecuyer1988>::param_oi_tidx)
    ;
} 

