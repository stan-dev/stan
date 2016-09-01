data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
}

transformed data {
  vector[d_int] transformed_data_vector;
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix = mdivide_left_spd(d_matrix,d_matrix);
  transformed_data_vector = mdivide_left_spd(d_matrix,d_vector);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  vector[d_int] transformed_param_vector;
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix = mdivide_left_spd(d_matrix,d_matrix);
  transformed_param_vector = mdivide_left_spd(d_matrix,d_vector);
  transformed_param_matrix = mdivide_left_spd(p_matrix,d_matrix);
  transformed_param_vector = mdivide_left_spd(p_matrix,d_vector);
  transformed_param_matrix = mdivide_left_spd(d_matrix,p_matrix);
  transformed_param_vector = mdivide_left_spd(d_matrix,p_vector);
  transformed_param_matrix = mdivide_left_spd(p_matrix,p_matrix);
  transformed_param_vector = mdivide_left_spd(p_matrix,p_vector);
}
model {  
}
