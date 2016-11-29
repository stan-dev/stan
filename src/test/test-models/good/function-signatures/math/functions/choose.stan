data { 
  int d_int;
  int r_int;
  
}
transformed data {
  int transformed_data_int;

  transformed_data_int <- choose(r_int, d_int);
}
parameters {
  real y_p;
}
model {  
  y_p ~ normal(0,1);
}
