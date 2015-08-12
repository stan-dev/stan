data { 
  int d_int;
  int d_int_array[d_int];
  real d_real;
  real d_real_array[d_int];
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- bernoulli_logit_log(d_int, d_int);
  transformed_data_real <- bernoulli_logit_log(d_int, d_real);
  transformed_data_real <- bernoulli_logit_log(d_int, d_vector);
  transformed_data_real <- bernoulli_logit_log(d_int, d_row_vector);
  transformed_data_real <- bernoulli_logit_log(d_int, d_real_array);
  transformed_data_real <- bernoulli_logit_log(d_int_array, d_int);
  transformed_data_real <- bernoulli_logit_log(d_int_array, d_real);
  transformed_data_real <- bernoulli_logit_log(d_int_array, d_vector);
  transformed_data_real <- bernoulli_logit_log(d_int_array, d_row_vector);
  transformed_data_real <- bernoulli_logit_log(d_int_array, d_real_array);
}
parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- bernoulli_logit_log(d_int, d_int);
  transformed_param_real <- bernoulli_logit_log(d_int, d_real);
  transformed_param_real <- bernoulli_logit_log(d_int, p_real);
  transformed_param_real <- bernoulli_logit_log(d_int, d_vector);
  transformed_param_real <- bernoulli_logit_log(d_int, p_vector);
  transformed_param_real <- bernoulli_logit_log(d_int, d_row_vector);
  transformed_param_real <- bernoulli_logit_log(d_int, p_row_vector);
  transformed_param_real <- bernoulli_logit_log(d_int, d_real_array);
  transformed_param_real <- bernoulli_logit_log(d_int, p_real_array);
  transformed_param_real <- bernoulli_logit_log(d_int_array, d_int);
  transformed_param_real <- bernoulli_logit_log(d_int_array, d_real);
  transformed_param_real <- bernoulli_logit_log(d_int_array, p_real);
  transformed_param_real <- bernoulli_logit_log(d_int_array, d_vector);
  transformed_param_real <- bernoulli_logit_log(d_int_array, p_vector);
  transformed_param_real <- bernoulli_logit_log(d_int_array, d_row_vector);
  transformed_param_real <- bernoulli_logit_log(d_int_array, p_row_vector);
  transformed_param_real <- bernoulli_logit_log(d_int_array, d_real_array);
  transformed_param_real <- bernoulli_logit_log(d_int_array, p_real_array);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
