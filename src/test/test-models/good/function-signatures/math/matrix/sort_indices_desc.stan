data {
  int d_int;
  array[d_int] int d_int_array;
  array[d_int] real d_real_array;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  array[d_int] int transformed_data_int_array;
  array[d_int] real transformed_data_real_array;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;
  transformed_data_int_array = sort_indices_desc(d_int_array);
  transformed_data_int_array = sort_indices_desc(d_real_array);
  transformed_data_int_array = sort_indices_desc(d_vector);
  transformed_data_int_array = sort_indices_desc(d_row_vector);
}
parameters {
  array[d_int] real p_real_array;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  array[d_int] real transformed_param_real_array;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;
  {
    array[d_int] int local_int_array;
    local_int_array = sort_indices_desc(p_real_array);
    local_int_array = sort_indices_desc(p_vector);
    local_int_array = sort_indices_desc(p_row_vector);
    local_int_array = sort_indices_desc(transformed_param_real_array);
    local_int_array = sort_indices_desc(transformed_param_vector);
    local_int_array = sort_indices_desc(transformed_param_row_vector);
  }
}
model {
  {
    array[d_int] int local_int_array;
    local_int_array = sort_indices_desc(d_int_array);
    local_int_array = sort_indices_desc(d_real_array);
    local_int_array = sort_indices_desc(d_vector);
    local_int_array = sort_indices_desc(d_row_vector);
    local_int_array = sort_indices_desc(transformed_data_int_array);
    local_int_array = sort_indices_desc(transformed_data_real_array);
    local_int_array = sort_indices_desc(transformed_data_vector);
    local_int_array = sort_indices_desc(transformed_data_row_vector);
    local_int_array = sort_indices_desc(p_real_array);
    local_int_array = sort_indices_desc(p_vector);
    local_int_array = sort_indices_desc(p_row_vector);
    local_int_array = sort_indices_desc(transformed_param_real_array);
    local_int_array = sort_indices_desc(transformed_param_vector);
    local_int_array = sort_indices_desc(transformed_param_row_vector);
  }
}

