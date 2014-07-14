data { 
  int d_int;
  int e_int;
  int f_int;
  real d_real;
  real e_real;
  real f_real;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

   transformed_data_real  <- fma(d_int, e_int, f_int );
   transformed_data_real <- fma(d_real, d_int, e_int);   
   transformed_data_real <- fma(d_real, e_real, e_int);
   transformed_data_real <- fma(d_real, d_int, e_real);
   transformed_data_real <- fma(d_real, e_real, f_real);
   transformed_data_real <- fma(d_int, d_real, e_int);
   transformed_data_real <- fma(d_int, d_real, e_real);
   transformed_data_real <- fma(d_int, e_int, d_real);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

   transformed_param_real <- fma(d_int, e_int, f_int );
   transformed_param_real <- fma(d_real, d_int, e_int);
   transformed_param_real <- fma(d_real, e_real, e_int);
   transformed_param_real <- fma(d_real, d_int, e_real);
   transformed_param_real <- fma(d_real, e_real, f_real);
   transformed_param_real <- fma(d_int, d_real, e_int);
   transformed_param_real <- fma(d_int, d_real, e_real);
   transformed_param_real <- fma(d_int, e_int, d_real);

   transformed_param_real <- fma(p_real, p_real, p_real );
   transformed_param_real <- fma(p_real, p_real, d_real );
   transformed_param_real <- fma(p_real, p_real, d_int );
   transformed_param_real <- fma(p_real, d_real, p_real);
   transformed_param_real <- fma(p_real, e_real, p_real);
   transformed_param_real <- fma(p_real, e_real, d_int);
   transformed_param_real <- fma(p_real, d_int, p_real);
   transformed_param_real <- fma(p_real, d_int, e_real);
   transformed_param_real <- fma(p_real, d_int, e_int);
   transformed_param_real <- fma(d_real, p_real, p_real);
   transformed_param_real <- fma(d_real, p_real, d_real);
   transformed_param_real <- fma(d_real, p_real, d_int);
   transformed_param_real <- fma(d_real, e_real, p_real);
   transformed_param_real <- fma(d_real, d_int, p_real);
   transformed_param_real <- fma(d_real, e_real, p_real);
   transformed_param_real <- fma(d_int, p_real, p_real);
   transformed_param_real <- fma(d_int, p_real, d_real);
   transformed_param_real <- fma(d_int, p_real, e_int);
   transformed_param_real <- fma(d_int, d_real, p_real);
   transformed_param_real <- fma(d_int, e_int, p_real);
}
model {  
  y_p ~ normal(0,1);
}

   
   









   
  
  