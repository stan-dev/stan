data { 
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;

  vector[2] x3y[3];
  row_vector[2] x4y[3];
  matrix[2,3] x5y[3];

  int x1z[3,4];
  real x2z[3,4];
  vector[2] x3z[3,4];
  row_vector[2] x4z[3,4];
  matrix[2,3] x5z[3,4];

  int x1w[3,4,5];
  real x2w[3,4,5];
  vector[2] x3w[3,4,5];
  row_vector[2] x4w[3,4,5];
  matrix[2,3] x5w[3,4,5];
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  vector[2] trans_x3y[3];
  row_vector[2] trans_x4y[3];
  matrix[2,3] trans_x5y[3];

  real trans_x2z[3,4];
  vector[2] trans_x3z[3,4];
  row_vector[2] trans_x4z[3,4];
  matrix[2,3] trans_x5z[3,4];

  real trans_x2w[3,4,5];
  vector[2] trans_x3w[3,4,5];
  row_vector[2] trans_x4w[3,4,5];
  matrix[2,3] trans_x5w[3,4,5];

  transformed_data_matrix <- inv_sqrt(d_matrix);
  transformed_data_vector <- inv_sqrt(d_vector);
  transformed_data_row_vector <- inv_sqrt(d_row_vector);
  trans_x3y <- inv_sqrt(x3y);
  trans_x4y <- inv_sqrt(x4y);
  trans_x5y <- inv_sqrt(x5y);

  trans_x2z <- inv_sqrt(x1z);
  trans_x2z <- inv_sqrt(x2z);
  trans_x3z <- inv_sqrt(x3z);
  trans_x4z <- inv_sqrt(x4z);
  trans_x5z <- inv_sqrt(x5z);

  trans_x2w <- inv_sqrt(x1w);
  trans_x2w <- inv_sqrt(x2w);
  trans_x3w <- inv_sqrt(x3w);
  trans_x4w <- inv_sqrt(x4w);
  trans_x5w <- inv_sqrt(x5w);
}
parameters {
  real p_real;
  real y_p;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;

  vector[2] p_x3y[3];
  row_vector[2] p_x4y[3];
  matrix[2,3] p_x5y[3];

  real p_x2z[3,4];
  vector[2] p_x3z[3,4];
  row_vector[2] p_x4z[3,4];
  matrix[2,3] p_x5z[3,4];

  real p_x2w[3,4,5];
  vector[2] p_x3w[3,4,5];
  row_vector[2] p_x4w[3,4,5];
  matrix[2,3] p_x5w[3,4,5];
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;
  vector[2] trans_p_x3y[3];
  row_vector[2] trans_p_x4y[3];
  matrix[2,3] trans_p_x5y[3];

  real trans_p_x2z[3,4];
  vector[2] trans_p_x3z[3,4];
  row_vector[2] trans_p_x4z[3,4];
  matrix[2,3] trans_p_x5z[3,4];

  real trans_p_x2w[3,4,5];
  vector[2] trans_p_x3w[3,4,5];
  row_vector[2] trans_p_x4w[3,4,5];
  matrix[2,3] trans_p_x5w[3,4,5];

  transformed_param_matrix <- inv_sqrt(d_matrix);
  transformed_param_vector <- inv_sqrt(d_vector);
  transformed_param_row_vector <- inv_sqrt(d_row_vector);
  transformed_param_matrix <- inv_sqrt(p_matrix);
  transformed_param_vector <- inv_sqrt(p_vector);
  transformed_param_row_vector <- inv_sqrt(p_row_vector);

  trans_p_x3y <- inv_sqrt(p_x3y);
  trans_p_x4y <- inv_sqrt(p_x4y);
  trans_p_x5y <- inv_sqrt(p_x5y);

  trans_p_x2z <- inv_sqrt(p_x2z);
  trans_p_x3z <- inv_sqrt(p_x3z);
  trans_p_x4z <- inv_sqrt(p_x4z);
  trans_p_x5z <- inv_sqrt(p_x5z);

  trans_p_x2w <- inv_sqrt(p_x2w);
  trans_p_x3w <- inv_sqrt(p_x3w);
  trans_p_x4w <- inv_sqrt(p_x4w);
  trans_p_x5w <- inv_sqrt(p_x5w);
}
model {  
  y_p ~ normal(0,1);
}
