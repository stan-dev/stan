RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::rstan<%model_name%_namespace::%model_name%> >("%model_name%")
    // .constructor<Rcpp::List>() 
    .constructor() 
    .method("nuts_command", &rstan::rstan<%model_name%_namespace::%model_name%>::nuts_command)
    ;
}
