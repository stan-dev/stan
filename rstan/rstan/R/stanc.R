stanc <- function(stan_model_code, model_name = "anon_model") {
  .Call("stanc", stan_model_code, model_name, PACKAGE = "rstan");
}


version <- function() {
  .Call("version", PACKAGE = "rstan");
}

## test examples 
#   stanc("hello", "hellomodel")

#   stanmodel <- "
#   data {
#     int(0,) N; 
#     real y[N];
#   } 
#   parameters {
#     real mu;
#   } 
#   model {
#     y ~ normal(mu, 1); 
#   } 
#   "

#   a <- stanc(stanmodel, 'stdnormal')
#   cat("stan ver: ", version(), sep = ''); 
