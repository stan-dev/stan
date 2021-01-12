data {
  int d_int;
  array[d_int] int d_int_array;
  real d_real;
  array[d_int] real d_real_array;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_int, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_int, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_int, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real_array,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real_array,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real_array,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real_array,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_real_array,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_vector, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_vector, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_row_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_row_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_row_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_row_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real, d_row_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_int,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_int,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_int,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_real,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_real,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_real,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_real_array, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_real_array, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_real_array, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_real_array, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_real_array, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_row_vector, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_row_vector, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_row_vector, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_row_vector, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_real_array,
                                            d_row_vector, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_int, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_int, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_int,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real_array,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real_array,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real_array,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real_array,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_real_array,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_row_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_row_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_row_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_row_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_vector, d_row_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_int,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_int,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_int,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_real,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_real,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_real,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_real_array, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_real_array, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_real_array, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_real_array, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_real_array, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_row_vector, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_row_vector, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_row_vector, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_row_vector, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real, d_row_vector,
                                            d_row_vector, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_int,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_int,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_int,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_real,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_real,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_real,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_real_array, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_real_array, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_real_array, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_real_array, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_real_array, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_row_vector, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_row_vector, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_row_vector, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_row_vector, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_int,
                                            d_row_vector, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_int,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_int,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_int,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_int,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_int,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_real,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_real,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_real,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_real,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_real,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_real_array, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_real_array, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_real_array, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_real_array, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_real_array, d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_vector,
                                            d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_vector,
                                            d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_vector,
                                            d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_vector,
                                            d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real, d_vector,
                                            d_row_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_row_vector, d_int);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_row_vector, d_real);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_row_vector, d_real_array);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_row_vector, d_vector);
  transformed_data_real = pareto_type_2_cdf(d_real_array, d_real,
                                            d_row_vector, d_row_vector);
}
parameters {
  real p_real;
  array[d_int] real p_real_array;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real y_p;
}
transformed parameters {
  real transformed_param_real;
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_real_array,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_vector,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_int, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, d_row_vector,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, d_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real, p_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_int, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_real_array,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_vector,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_int, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_real_array, p_row_vector,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_real_array,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, d_row_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_real_array,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_int, p_row_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_real_array,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, d_row_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_real_array,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real, p_row_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             d_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array, p_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_real_array,
                                             p_row_vector, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_int,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_real,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_real_array, p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             d_int);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             d_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             d_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             d_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             d_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             p_real);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             p_real_array);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             p_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector, d_vector,
                                             p_row_vector);
  transformed_param_real = pareto_type_2_cdf(d_vector, d_vector,
                                             d_row_vector, d_int);
}
model {
  y_p ~ normal(0, 1);
}

