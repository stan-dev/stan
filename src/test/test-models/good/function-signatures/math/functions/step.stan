data {
  int d_int;
  real d_real;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

   transformed_data_real <- step(d_int);
   transformed_data_real <- step(d_real);
}
parameters {
  real p_real;
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = step(d_int);
  transformed_param_real = step(d_real);
  transformed_param_real = step(p_real);

  transformed_param_real = step(rows(d_vector));
  transformed_param_real = step(rows(p_vector));
  transformed_param_real = step(rows(d_row_vector));
  transformed_param_real = step(rows(p_row_vector));
  transformed_param_real = step(rows(d_matrix));
  transformed_param_real = step(rows(p_matrix));

  transformed_param_real = step(cols(d_vector));
  transformed_param_real = step(cols(p_vector));
  transformed_param_real = step(cols(d_row_vector));
  transformed_param_real = step(cols(p_row_vector));
  transformed_param_real = step(cols(d_matrix));
  transformed_param_real = step(cols(p_matrix));
}
model {
  y_p ~ normal(0,1);
}
