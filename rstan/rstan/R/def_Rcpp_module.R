def_Rcpp_module <- function(model_name) {
  src <- '
RCPP_MODULE(%model_name%){
  Rcpp::class_<rstan::rstan<%model_name%_namespace::%model_name%> >("%model_name%")
    .constructor<Rcpp::List>() 
    .method("nuts_command", &rstan::rstan<%model_name%_namespace::%model_name%>::nuts_command)
    ;
}
' 
  gsub("%model_name%", model_name, src); 

} 

