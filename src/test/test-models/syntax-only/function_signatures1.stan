data { 
  int d_int;
  int d_int_array[d_int];
  int d_int_array_2[d_int, d_int];
  int d_int_array_3[d_int, d_int, d_int];
  real d_real;
  real d_real_array[d_int];
  real d_real_array_2[d_int, d_int];
  real d_real_array_3[d_int, d_int, d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  int transformed_data_int_array[d_int];
  real transformed_data_real;
  real transformed_data_real_array[d_int];
  real transformed_data_real_array_2[d_int, d_int];
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  //*** Matrix Operations ***

  // Linear Algebra Functions and Scalars
  //   matrix division infix operators
  transformed_data_row_vector <- d_row_vector / d_matrix;
  transformed_data_vector <- d_matrix \ d_vector;
  
  //   linear algebra functions
  transformed_data_real <- trace(d_matrix);
  transformed_data_real <- determinant(d_matrix);
  transformed_data_real <- log_determinant(d_matrix);
  transformed_data_matrix <- inverse(d_matrix);
  transformed_data_matrix <- inverse_spd(d_matrix);
  transformed_data_vector <- eigenvalues_sym(d_matrix);
  transformed_data_matrix <- eigenvectors_sym(d_matrix);
  transformed_data_matrix <- cholesky_decompose(d_matrix);
  transformed_data_vector <- singular_values(d_matrix);
  transformed_data_matrix <- qr_Q(d_matrix);
  transformed_data_matrix <- qr_R(d_matrix);
}
parameters {
  real p_real;
  real p_real_array[d_int];
  real p_real_array_2[d_int, d_int];
  real p_real_array_3[d_int, d_int, d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;
  real transformed_param_real_array[d_int];
  real transformed_param_real_array_2[d_int, d_int];
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  //*** Real-Valued Basic Functions ***
  // mathematical constants
  transformed_param_real <- pi();
  transformed_param_real <- e();
  transformed_param_real <- sqrt2();
  transformed_param_real <- log2();
  transformed_param_real <- log10();

  // special values
  transformed_param_real <- not_a_number();
  transformed_param_real <- positive_infinity();
  transformed_param_real <- negative_infinity();
  transformed_param_real <- machine_precision();

  //   link functions
  transformed_param_real <- logit(d_real);
  transformed_param_real <- logit(p_real);
  transformed_param_real <- inv_logit(d_real);
  transformed_param_real <- inv_logit(p_real);
  transformed_param_real <- inv_cloglog(d_real);
  transformed_param_real <- inv_cloglog(p_real);

  //   probability-related functions
  transformed_param_real <- erf(d_real);
  transformed_param_real <- erf(p_real);
  transformed_param_real <- erfc(d_real);
  transformed_param_real <- erfc(p_real);

  //   combinatorial functions
  transformed_param_real <- tgamma(d_real);
  transformed_param_real <- tgamma(p_real);

  //   composed functions
  transformed_param_real <- expm1(d_real);
  transformed_param_real <- expm1(p_real);

  //*** Array Operations ***
  transformed_param_real <- sum(d_real_array);
  transformed_param_real <- sum(p_real_array);
  transformed_param_real <- mean(d_real_array);
  transformed_param_real <- mean(p_real_array);
  transformed_param_real <- variance(d_real_array);
  transformed_param_real <- variance(p_real_array);
  transformed_param_real <- sd(d_real_array);
  transformed_param_real <- sd(p_real_array);
  transformed_param_real <- log_sum_exp(d_real_array);
  transformed_param_real <- log_sum_exp(p_real_array);

  //*** Array to Array, Vec to Vec Operations ***
  transformed_param_real_array <- cumulative_sum(d_real_array);
  transformed_param_real_array <- cumulative_sum(p_real_array);
  transformed_param_vector <- cumulative_sum(d_vector);
  transformed_param_vector <- cumulative_sum(p_vector);
  transformed_param_row_vector <- cumulative_sum(d_row_vector);
  transformed_param_row_vector <- cumulative_sum(p_row_vector);


  //*** Matrix Operations ***
  // Integer-Valued Matrix Size Functions


  // Matrix Arithmetic Operators
  //   negation prefix operators
  transformed_param_vector <- -d_vector;
  transformed_param_vector <- -p_vector;
  transformed_param_row_vector <- -d_row_vector;
  transformed_param_row_vector <- -p_row_vector;
  transformed_param_matrix <- -d_matrix;
  transformed_param_matrix <- -p_matrix;

  //   infix matrix operators
  transformed_param_vector <- d_vector + d_vector;
  transformed_param_vector <- p_vector + d_vector;
  transformed_param_vector <- d_vector + p_vector;
  transformed_param_vector <- p_vector + p_vector;
  transformed_param_row_vector <- d_row_vector + d_row_vector;
  transformed_param_row_vector <- p_row_vector + d_row_vector;
  transformed_param_row_vector <- d_row_vector + p_row_vector;
  transformed_param_row_vector <- p_row_vector + p_row_vector;
  transformed_param_matrix <- d_matrix + d_matrix;
  transformed_param_matrix <- p_matrix + d_matrix;
  transformed_param_matrix <- d_matrix + p_matrix;
  transformed_param_matrix <- p_matrix + p_matrix;
  transformed_param_vector <- d_vector - d_vector;
  transformed_param_vector <- p_vector - d_vector;
  transformed_param_vector <- d_vector - p_vector;
  transformed_param_vector <- p_vector - p_vector;
  transformed_param_row_vector <- d_row_vector - d_row_vector;
  transformed_param_row_vector <- p_row_vector - d_row_vector;
  transformed_param_row_vector <- d_row_vector - p_row_vector;
  transformed_param_row_vector <- p_row_vector - p_row_vector;
  transformed_param_matrix <- d_matrix - d_matrix;
  transformed_param_matrix <- p_matrix - d_matrix;
  transformed_param_matrix <- d_matrix - p_matrix;
  transformed_param_matrix <- p_matrix - p_matrix;
  transformed_param_vector <- d_real * d_vector;
  transformed_param_vector <- p_real * d_vector;
  transformed_param_vector <- d_real * p_vector;
  transformed_param_vector <- p_real * p_vector;
  transformed_param_row_vector <- d_real * d_row_vector;
  transformed_param_row_vector <- p_real * d_row_vector;
  transformed_param_row_vector <- d_real * p_row_vector;
  transformed_param_row_vector <- p_real * p_row_vector;
  transformed_param_matrix <- d_real * d_matrix;
  transformed_param_matrix <- p_real * d_matrix;
  transformed_param_matrix <- d_real * p_matrix;
  transformed_param_matrix <- p_real * p_matrix;
  transformed_param_vector <- d_vector * d_real;
  transformed_param_vector <- p_vector * d_real;
  transformed_param_vector <- d_vector * p_real;
  transformed_param_vector <- p_vector * p_real;
  transformed_param_row_vector <- d_row_vector * d_real;
  transformed_param_row_vector <- p_row_vector * d_real;
  transformed_param_row_vector <- d_row_vector * p_real;
  transformed_param_row_vector <- p_row_vector * p_real;
  transformed_param_matrix <- d_matrix * d_real;
  transformed_param_matrix <- p_matrix * d_real;
  transformed_param_matrix <- d_matrix * p_real;
  transformed_param_matrix <- p_matrix * p_real;
  transformed_param_real <- d_row_vector * d_vector;
  transformed_param_real <- p_row_vector * d_vector;
  transformed_param_real <- d_row_vector * p_vector;
  transformed_param_real <- p_row_vector * p_vector;
  transformed_param_row_vector <- d_row_vector * d_matrix;
  transformed_param_row_vector <- p_row_vector * d_matrix;
  transformed_param_row_vector <- d_row_vector * p_matrix;
  transformed_param_row_vector <- p_row_vector * p_matrix;
  transformed_param_matrix <- d_matrix * d_real;
  transformed_param_matrix <- p_matrix * d_real;
  transformed_param_matrix <- d_matrix * p_real;
  transformed_param_matrix <- p_matrix * p_real;
  transformed_param_vector <- d_matrix * d_vector;
  transformed_param_vector <- p_matrix * d_vector;
  transformed_param_vector <- d_matrix * p_vector;
  transformed_param_vector <- p_matrix * p_vector;
  transformed_param_matrix <- d_matrix * d_matrix;
  transformed_param_matrix <- p_matrix * d_matrix;
  transformed_param_matrix <- d_matrix * p_matrix;
  transformed_param_matrix <- p_matrix * p_matrix;

  //   broadcast infix operators
  transformed_param_vector <- d_vector + d_real;
  transformed_param_vector <- p_vector + d_real;
  transformed_param_vector <- d_vector + p_real;
  transformed_param_vector <- p_vector + p_real;
  transformed_param_vector <- d_real + d_vector;
  transformed_param_vector <- p_real + d_vector; 
  transformed_param_vector <- d_real + p_vector;
  transformed_param_vector <- p_real + p_vector;
  transformed_param_row_vector <- d_row_vector + d_real;
  transformed_param_row_vector <- p_row_vector + d_real;
  transformed_param_row_vector <- d_row_vector + p_real;
  transformed_param_row_vector <- p_row_vector + p_real;
  transformed_param_row_vector <- d_real + d_row_vector;
  transformed_param_row_vector <- p_real + d_row_vector; 
  transformed_param_row_vector <- d_real + p_row_vector;
  transformed_param_row_vector <- p_real + p_row_vector;
  transformed_param_matrix <- d_matrix + d_real;
  transformed_param_matrix <- p_matrix + d_real;
  transformed_param_matrix <- d_matrix + p_real;
  transformed_param_matrix <- p_matrix + p_real;
  transformed_param_matrix <- d_real + d_matrix;
  transformed_param_matrix <- p_real + d_matrix; 
  transformed_param_matrix <- d_real + p_matrix;
  transformed_param_matrix <- p_real + p_matrix;
  transformed_param_vector <- d_vector - d_real;
  transformed_param_vector <- p_vector - d_real;
  transformed_param_vector <- d_vector - p_real;
  transformed_param_vector <- p_vector - p_real;
  transformed_param_vector <- d_real - d_vector;
  transformed_param_vector <- p_real - d_vector; 
  transformed_param_vector <- d_real - p_vector;
  transformed_param_vector <- p_real - p_vector;
  transformed_param_row_vector <- d_row_vector - d_real;
  transformed_param_row_vector <- p_row_vector - d_real;
  transformed_param_row_vector <- d_row_vector - p_real;
  transformed_param_row_vector <- p_row_vector - p_real;
  transformed_param_row_vector <- d_real - d_row_vector;
  transformed_param_row_vector <- p_real - d_row_vector; 
  transformed_param_row_vector <- d_real - p_row_vector;
  transformed_param_row_vector <- p_real - p_row_vector;
  transformed_param_matrix <- d_matrix - d_real;
  transformed_param_matrix <- p_matrix - d_real;
  transformed_param_matrix <- d_matrix - p_real;
  transformed_param_matrix <- p_matrix - p_real;
  transformed_param_matrix <- d_real - d_matrix;
  transformed_param_matrix <- p_real - d_matrix; 
  transformed_param_matrix <- d_real - p_matrix;
  transformed_param_matrix <- p_real - p_matrix;
  
  //   elementwise products
  transformed_param_vector <- d_vector .* d_vector;
  transformed_param_vector <- p_vector .* d_vector;
  transformed_param_vector <- d_vector .* p_vector;
  transformed_param_vector <- p_vector .* p_vector;
  transformed_param_row_vector <- d_row_vector .* d_row_vector;
  transformed_param_row_vector <- p_row_vector .* d_row_vector;
  transformed_param_row_vector <- d_row_vector .* p_row_vector;
  transformed_param_row_vector <- p_row_vector .* p_row_vector;
  transformed_param_matrix <- d_matrix .* d_matrix;
  transformed_param_matrix <- p_matrix .* d_matrix;
  transformed_param_matrix <- d_matrix .* p_matrix;
  transformed_param_matrix <- p_matrix .* p_matrix;
  transformed_param_vector <- d_vector ./ d_vector;
  transformed_param_vector <- p_vector ./ d_vector;
  transformed_param_vector <- d_vector ./ p_vector;
  transformed_param_vector <- p_vector ./ p_vector;
  transformed_param_row_vector <- d_row_vector ./ d_row_vector;
  transformed_param_row_vector <- p_row_vector ./ d_row_vector;
  transformed_param_row_vector <- d_row_vector ./ p_row_vector;
  transformed_param_row_vector <- p_row_vector ./ p_row_vector;
  transformed_param_matrix <- d_matrix ./ d_matrix;
  transformed_param_matrix <- p_matrix ./ d_matrix;
  transformed_param_matrix <- d_matrix ./ p_matrix;
  transformed_param_matrix <- p_matrix ./ p_matrix;

  //   elementwise logarithms
  transformed_param_vector <- log(d_vector);
  transformed_param_vector <- log(p_vector);
  transformed_param_row_vector <- log(d_row_vector);
  transformed_param_row_vector <- log(p_row_vector);
  transformed_param_matrix <- log(d_matrix);
  transformed_param_matrix <- log(p_matrix);
  transformed_param_vector <- exp(d_vector);
  transformed_param_vector <- exp(p_vector);
  transformed_param_row_vector <- exp(d_row_vector);
  transformed_param_row_vector <- exp(p_row_vector);
  transformed_param_matrix <- exp(d_matrix);
  transformed_param_matrix <- exp(p_matrix);

  //  dot products
  transformed_param_real <- dot_product(d_vector, d_vector);
  transformed_param_real <- dot_product(p_vector, d_vector);
  transformed_param_real <- dot_product(d_vector, p_vector);
  transformed_param_real <- dot_product(p_vector, p_vector);
  transformed_param_real <- dot_product(d_vector, d_row_vector);
  transformed_param_real <- dot_product(p_vector, d_row_vector);
  transformed_param_real <- dot_product(d_vector, p_row_vector);
  transformed_param_real <- dot_product(p_vector, p_row_vector);
  transformed_param_real <- dot_product(d_row_vector, d_vector);
  transformed_param_real <- dot_product(p_row_vector, d_vector);
  transformed_param_real <- dot_product(d_row_vector, p_vector);
  transformed_param_real <- dot_product(p_row_vector, p_vector);
  transformed_param_real <- dot_product(d_row_vector, d_row_vector);
  transformed_param_real <- dot_product(p_row_vector, d_row_vector);
  transformed_param_real <- dot_product(d_row_vector, p_row_vector);
  transformed_param_real <- dot_product(p_row_vector, p_row_vector);

  transformed_param_real <- dot_self(p_vector);
  transformed_param_real <- dot_self(p_row_vector);  

  // quadratic forms
  transformed_param_real <- quad_form(d_matrix,d_vector);
  transformed_param_real <- quad_form(d_matrix,p_vector);
  transformed_param_real <- quad_form(p_matrix,d_vector);
  transformed_param_real <- quad_form(p_matrix,p_vector);
  transformed_param_matrix <- quad_form(d_matrix,d_matrix);
  transformed_param_matrix <- quad_form(d_matrix,p_matrix);
  transformed_param_matrix <- quad_form(p_matrix,d_matrix);
  transformed_param_matrix <- quad_form(p_matrix,p_matrix);
  transformed_param_real <- quad_form_sym(d_matrix,d_vector);
  transformed_param_real <- quad_form_sym(d_matrix,p_vector);
  transformed_param_real <- quad_form_sym(p_matrix,d_vector);
  transformed_param_real <- quad_form_sym(p_matrix,p_vector);
  transformed_param_matrix <- quad_form_sym(d_matrix,d_matrix);
  transformed_param_matrix <- quad_form_sym(d_matrix,p_matrix);
  transformed_param_matrix <- quad_form_sym(p_matrix,d_matrix);
  transformed_param_matrix <- quad_form_sym(p_matrix,p_matrix);
  transformed_param_real <- trace_quad_form(d_matrix,d_vector);
  transformed_param_real <- trace_quad_form(d_matrix,p_vector);
  transformed_param_real <- trace_quad_form(p_matrix,d_vector);
  transformed_param_real <- trace_quad_form(p_matrix,p_vector);
  transformed_param_real <- trace_quad_form(d_matrix,d_matrix);
  transformed_param_real <- trace_quad_form(d_matrix,p_matrix);
  transformed_param_real <- trace_quad_form(p_matrix,d_matrix);
  transformed_param_real <- trace_quad_form(p_matrix,p_matrix);
  transformed_param_real <- trace_gen_quad_form(d_matrix,d_matrix,d_matrix);
  transformed_param_real <- trace_gen_quad_form(d_matrix,d_matrix,p_matrix);
  transformed_param_real <- trace_gen_quad_form(d_matrix,p_matrix,d_matrix);
  transformed_param_real <- trace_gen_quad_form(p_matrix,d_matrix,d_matrix);
  transformed_param_real <- trace_gen_quad_form(p_matrix,p_matrix,d_matrix);
  transformed_param_real <- trace_gen_quad_form(p_matrix,d_matrix,p_matrix);
  transformed_param_real <- trace_gen_quad_form(d_matrix,p_matrix,p_matrix);
  transformed_param_real <- trace_gen_quad_form(p_matrix,p_matrix,p_matrix);


  //  reductions
  transformed_param_real <- min(d_vector);
  transformed_param_real <- min(p_vector);
  transformed_param_real <- min(d_row_vector);
  transformed_param_real <- min(p_row_vector);
  transformed_param_real <- min(d_matrix);
  transformed_param_real <- min(p_matrix);
  transformed_param_real <- max(d_vector);
  transformed_param_real <- max(p_vector);
  transformed_param_real <- max(d_row_vector);
  transformed_param_real <- max(p_row_vector);
  transformed_param_real <- max(d_matrix);
  transformed_param_real <- max(p_matrix);

  //  sums and products
  transformed_param_real <- sum(d_vector);
  transformed_param_real <- sum(p_vector);
  transformed_param_real <- sum(d_row_vector);
  transformed_param_real <- sum(p_row_vector);
  transformed_param_real <- sum(d_matrix);
  transformed_param_real <- sum(p_matrix);
  transformed_param_real <- prod(d_vector);
  transformed_param_real <- prod(p_vector);
  transformed_param_real <- prod(d_row_vector);
  transformed_param_real <- prod(p_row_vector);
  transformed_param_real <- prod(d_matrix);
  transformed_param_real <- prod(p_matrix);

  //  sample moments
  transformed_param_real <- mean(d_vector);
  transformed_param_real <- mean(p_vector);
  transformed_param_real <- mean(d_row_vector);
  transformed_param_real <- mean(p_row_vector);
  transformed_param_real <- mean(d_matrix);
  transformed_param_real <- mean(p_matrix);
  transformed_param_real <- variance(d_vector);
  transformed_param_real <- variance(p_vector);
  transformed_param_real <- variance(d_row_vector);
  transformed_param_real <- variance(p_row_vector);
  transformed_param_real <- variance(d_matrix);
  transformed_param_real <- variance(p_matrix);
  transformed_param_real <- sd(d_vector);
  transformed_param_real <- sd(p_vector);
  transformed_param_real <- sd(d_row_vector);
  transformed_param_real <- sd(p_row_vector);
  transformed_param_real <- sd(d_matrix);
  transformed_param_real <- sd(p_matrix);

  //Broadcast Functions
  transformed_param_vector <- rep_vector(d_real, d_int);
  transformed_param_vector <- rep_vector(p_real, d_int);
  transformed_param_row_vector <- rep_row_vector(d_real, d_int);
  transformed_param_row_vector <- rep_row_vector(p_real, d_int);
  transformed_param_matrix <- rep_matrix(d_real, d_int, d_int);
  transformed_param_matrix <- rep_matrix(p_real, d_int, d_int);
  transformed_param_matrix <- rep_matrix(d_vector, d_int);
  transformed_param_matrix <- rep_matrix(p_vector, d_int);
  transformed_param_matrix <- rep_matrix(d_row_vector, d_int);
  transformed_param_matrix <- rep_matrix(p_row_vector, d_int);

  //Containers Conversion Functions
  transformed_param_vector <- to_vector(d_matrix);
  transformed_param_vector <- to_vector(p_matrix);
  transformed_param_vector <- to_vector(d_vector);
  transformed_param_vector <- to_vector(p_vector);
  transformed_param_vector <- to_vector(d_row_vector);
  transformed_param_vector <- to_vector(p_row_vector);
  transformed_param_vector <- to_vector(d_int_array);
  transformed_param_vector <- to_vector(d_real_array);
  transformed_param_vector <- to_vector(p_real_array);

  transformed_param_row_vector <- to_row_vector(d_matrix);
  transformed_param_row_vector <- to_row_vector(p_matrix);
  transformed_param_row_vector <- to_row_vector(d_vector);
  transformed_param_row_vector <- to_row_vector(p_vector);
  transformed_param_row_vector <- to_row_vector(d_row_vector);
  transformed_param_row_vector <- to_row_vector(p_row_vector);
  transformed_param_row_vector <- to_row_vector(d_int_array);
  transformed_param_row_vector <- to_row_vector(d_real_array);
  transformed_param_row_vector <- to_row_vector(p_real_array);

  transformed_param_matrix <- to_matrix(d_matrix);
  transformed_param_matrix <- to_matrix(p_matrix);
  transformed_param_matrix <- to_matrix(d_vector);
  transformed_param_matrix <- to_matrix(p_vector);
  transformed_param_matrix <- to_matrix(d_row_vector);
  transformed_param_matrix <- to_matrix(p_row_vector);
  transformed_param_matrix <- to_matrix(d_int_array_2);
  transformed_param_matrix <- to_matrix(d_real_array_2);
  transformed_param_matrix <- to_matrix(p_real_array_2);

  transformed_param_real_array_2 <- to_array_2d(d_matrix);
  transformed_param_real_array_2 <- to_array_2d(p_matrix);

  transformed_param_real_array <- to_array_1d(d_matrix);
  transformed_param_real_array <- to_array_1d(p_matrix);
  transformed_param_real_array <- to_array_1d(d_vector);
  transformed_param_real_array <- to_array_1d(p_vector);
  transformed_param_real_array <- to_array_1d(d_row_vector);
  transformed_param_real_array <- to_array_1d(p_row_vector);
  transformed_param_real_array <- to_array_1d(d_real_array_2);
  transformed_param_real_array <- to_array_1d(p_real_array_2);
  transformed_param_real_array <- to_array_1d(d_real_array_3);
  transformed_param_real_array <- to_array_1d(p_real_array_3);  
  
  // Slice and Package Functions
  //   diagonal matrices
  transformed_param_vector <- diagonal(d_matrix);
  transformed_param_vector <- diagonal(p_matrix);
  transformed_param_matrix <- diag_matrix(d_vector);
  transformed_param_matrix <- diag_matrix(p_vector);
  transformed_param_vector <- col(d_matrix, d_int);
  transformed_param_vector <- col(p_matrix, d_int);
  transformed_param_row_vector <- row(d_matrix, d_int);
  transformed_param_row_vector <- row(p_matrix, d_int);

  //   transposition postfix operator
  transformed_param_matrix <- d_matrix';
  transformed_param_matrix <- p_matrix';  
  transformed_param_row_vector <- d_vector';
  transformed_param_row_vector <- p_vector';
  transformed_param_vector <- d_row_vector';
  transformed_param_vector <- p_row_vector';

  // Special Matrix Functions
  transformed_param_vector <- softmax(d_vector);
  transformed_param_vector <- softmax(p_vector);

  // Linear Algebra Functions and Solvers
  //   matrix division infix operators
  transformed_param_row_vector <- d_row_vector / d_matrix;
  transformed_param_row_vector <- p_row_vector / d_matrix;
  transformed_param_row_vector <- d_row_vector / p_matrix;
  transformed_param_row_vector <- p_row_vector / p_matrix;
  transformed_param_vector <- d_matrix \ d_vector;
  transformed_param_vector <- p_matrix \ d_vector;
  transformed_param_vector <- d_matrix \ p_vector;
  transformed_param_vector <- p_matrix \ p_vector;
  
  //   linear algebra functions
  transformed_param_real <- trace(d_matrix);
  transformed_param_real <- trace(p_matrix);
  transformed_param_real <- determinant(d_matrix);
  transformed_param_real <- determinant(p_matrix);
  transformed_param_real <- log_determinant(d_matrix);
  transformed_param_real <- log_determinant(p_matrix);
  transformed_param_matrix <- inverse(d_matrix);
  transformed_param_matrix <- inverse(p_matrix);
  transformed_param_matrix <- inverse_spd(d_matrix);
  transformed_param_matrix <- inverse_spd(p_matrix);
  transformed_param_vector <- eigenvalues_sym(d_matrix);
  transformed_param_vector <- eigenvalues_sym(p_matrix);
  transformed_param_matrix <- eigenvectors_sym(d_matrix);
  transformed_param_matrix <- eigenvectors_sym(p_matrix);
  transformed_param_matrix <- cholesky_decompose(d_matrix);
  transformed_param_matrix <- cholesky_decompose(p_matrix);
  transformed_param_vector <- singular_values(d_matrix);
  transformed_param_vector <- singular_values(p_matrix);
  transformed_param_matrix <- qr_Q(d_matrix);
  transformed_param_matrix <- qr_R(d_matrix);
  transformed_param_matrix <- qr_Q(p_matrix);
  transformed_param_matrix <- qr_R(p_matrix);
}
model {  
}
