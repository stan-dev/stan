data { 
  int d_int;
  int d_int_array[d_int];
  real d_real;
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;
  real transformed_data_real_array[d_int];
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  //*** Integer-Valued Basic Functions ***
  // integer-valued arithmetic operators
  //   binary infix operators
  transformed_data_int <- d_int + d_int;
  transformed_data_int <- d_int - d_int;
  transformed_data_int <- d_int * d_int;
  transformed_data_int <- d_int / d_int;

  //   unary prefix operators
  transformed_data_int <- -d_int;
  transformed_data_int <- +d_int;
  
  //   absolute functions
  transformed_data_int <- abs(d_int);
  transformed_data_int <- int_step(d_int);

  //   bound functions
  transformed_data_int <- min(d_int, d_int);
  transformed_data_int <- max(d_int, d_int);

  //*** Real-Valued Basic Functions ***
  // mathematical constants
  transformed_data_real <- pi();
  transformed_data_real <- e();
  transformed_data_real <- sqrt2();
  transformed_data_real <- log2();
  transformed_data_real <- log10();

  // special values
  transformed_data_real <- not_a_number();
  transformed_data_real <- positive_infinity();
  transformed_data_real <- negative_infinity();
  transformed_data_real <- machine_precision();

  // logical functions
  transformed_data_real <- if_else(d_int, d_real, d_real);
  transformed_data_real <- step(d_real);
  
  // real-valued arithmetic operators
  //   binary infix operators
  transformed_data_real <- d_real + d_real;
  transformed_data_real <- d_real - d_real;
  transformed_data_real <- d_real * d_real;
  transformed_data_real <- d_real / d_real;

  //   unary prefix operators
  transformed_data_real <- -d_real;
  transformed_data_real <- +d_real;

  //   absolute functions
  transformed_data_real <- abs(d_real);
  transformed_data_real <- fabs(d_real);
  transformed_data_real <- fdim(d_real, d_real);

  //   bounds functions
  transformed_data_real <- fmin(d_real, d_real);
  transformed_data_real <- fmax(d_real, d_real);

  //   arithmetic functions
  transformed_data_real <- fmod(d_real, d_real);

  //   rounding functions
  transformed_data_real <- floor(d_real);
  transformed_data_real <- ceil(d_real);
  transformed_data_real <- round(d_real);
  transformed_data_real <- trunc(d_real);

  //   power and logarithm functions
  transformed_data_real <- sqrt(d_real);
  transformed_data_real <- cbrt(d_real);
  transformed_data_real <- square(d_real);
  transformed_data_real <- exp(d_real);
  transformed_data_real <- exp2(d_real);
  transformed_data_real <- log(d_real);
  transformed_data_real <- log2(d_real);
  transformed_data_real <- log10(d_real);
  transformed_data_real <- pow(d_real, d_real);
  transformed_data_real <- inv(d_real);
  transformed_data_real <- inv_square(d_real);
  transformed_data_real <- inv_sqrt(d_real);


  //   trigonometric functions
  transformed_data_real <- hypot(d_real, d_real);
  transformed_data_real <- cos(d_real);
  transformed_data_real <- sin(d_real);
  transformed_data_real <- tan(d_real);
  transformed_data_real <- acos(d_real);
  transformed_data_real <- asin(d_real);
  transformed_data_real <- atan(d_real);
  transformed_data_real <- atan2(d_real, d_real);

  //   hyperbolic trigonometric functions
  transformed_data_real <- cosh(d_real);
  transformed_data_real <- sinh(d_real);
  transformed_data_real <- tanh(d_real);
  transformed_data_real <- acosh(d_real);
  transformed_data_real <- asinh(d_real);
  transformed_data_real <- atanh(d_real);

  //   link functions
  transformed_data_real <- logit(d_real);
  transformed_data_real <- inv_logit(d_real);
  transformed_data_real <- inv_cloglog(d_real);

  //   probability-related functions
  transformed_data_real <- erf(d_real);
  transformed_data_real <- erfc(d_real);
  transformed_data_real <- Phi(d_real);
  transformed_data_real <- Phi_approx(d_real);
  transformed_data_real <- binary_log_loss(d_int, d_real);
  transformed_data_real <- owens_t(d_real, d_real);

  //   combinatorial functions
  transformed_data_real <- tgamma(d_real);
  transformed_data_real <- lgamma(d_real);
  transformed_data_real <- digamma(d_real);
  transformed_data_real <- trigamma(d_real);
  transformed_data_real <- gamma_p(d_real, d_real);
  transformed_data_real <- gamma_q(d_real, d_real);
  transformed_data_real <- lmgamma(d_int, d_real);
  transformed_data_real <- lbeta(d_real, d_real);
  transformed_data_real <- binomial_coefficient_log(d_real, d_real);
  transformed_data_real <- bessel_first_kind(d_int, d_real);
  transformed_data_real <- bessel_second_kind(d_int, d_real);
  transformed_data_real <- modified_bessel_first_kind(d_int, d_real);
  transformed_data_real <- modified_bessel_second_kind(d_int, d_real);
  transformed_data_real <- falling_factorial(d_real, d_real);
  transformed_data_real <- rising_factorial(d_real, d_real);
  transformed_data_real <- log_falling_factorial(d_real, d_real);
  transformed_data_real <- log_rising_factorial(d_real, d_real);


  //   composed functions
  transformed_data_real <- expm1(d_real);
  transformed_data_real <- fma(d_real, d_real, d_real);
  transformed_data_real <- multiply_log(d_real, d_real);
  transformed_data_real <- log1p(d_real);
  transformed_data_real <- log1m(d_real);
  transformed_data_real <- log1p_exp(d_real);
  transformed_data_real <- log_sum_exp(d_real, d_real);
  transformed_data_real <- log_inv_logit(d_real);
  transformed_data_real <- log1m_inv_logit(d_real);

  //*** Array Operations ***
  transformed_data_real <- sum(d_real_array);
  transformed_data_int <- sum(d_int_array);
  transformed_data_real <- prod(d_real_array);
  transformed_data_int <- prod(d_int_array);
  transformed_data_real <- min(d_real_array);
  transformed_data_int <- min(d_int_array);
  transformed_data_real <- max(d_real_array);
  transformed_data_int <- max(d_int_array);
  transformed_data_real <- mean(d_real_array);
  transformed_data_real <- variance(d_real_array);
  transformed_data_real <- sd(d_real_array);
  transformed_data_real <- log_sum_exp(d_real_array);
  
  //*** Array to Array, Vec to Vec Operations ***
  transformed_data_real_array <- cumulative_sum(d_real_array);
  transformed_data_vector <- cumulative_sum(d_vector);
  transformed_data_row_vector <- cumulative_sum(d_row_vector);
  
  //*** Matrix Operations ***
  // Integer-Valued Matrix Size Functions
  transformed_data_int <- rows(d_vector);
  transformed_data_int <- rows(d_row_vector);
  transformed_data_int <- rows(d_matrix);
  transformed_data_int <- cols(d_vector);
  transformed_data_int <- cols(d_row_vector);
  transformed_data_int <- cols(d_matrix);

  // Matrix Arithmetic Operators
  //   negation prefix operators
  transformed_data_vector <- -d_vector;
  transformed_data_row_vector <- -d_row_vector;
  transformed_data_matrix <- -d_matrix;

  //   infix matrix operators
  transformed_data_vector <- d_vector + d_vector;
  transformed_data_row_vector <- d_row_vector + d_row_vector;
  transformed_data_matrix <- d_matrix + d_matrix;
  transformed_data_vector <- d_vector - d_vector;
  transformed_data_row_vector <- d_row_vector - d_row_vector;
  transformed_data_matrix <- d_matrix - d_matrix;
  transformed_data_vector <- d_real * d_vector;
  transformed_data_row_vector <- d_real * d_row_vector;
  transformed_data_matrix <- d_real * d_matrix;
  transformed_data_vector <- d_vector * d_real;
  transformed_data_row_vector <- d_row_vector * d_real;
  transformed_data_matrix <- d_matrix * d_real;
  transformed_data_real <- d_row_vector * d_vector;
  transformed_data_row_vector <- d_row_vector * d_matrix;
  transformed_data_matrix <- d_matrix * d_real;
  transformed_data_vector <- d_matrix * d_vector;
  transformed_data_matrix <- d_matrix * d_matrix;

  //   broadcast infix operators
  transformed_data_vector <- d_vector + d_real;
  transformed_data_vector <- d_real + d_vector;
  transformed_data_row_vector <- d_row_vector + d_real;
  transformed_data_row_vector <- d_real + d_row_vector;
  transformed_data_matrix <- d_matrix + d_real;
  transformed_data_matrix <- d_real + d_matrix;
  transformed_data_vector <- d_vector - d_real;
  transformed_data_vector <- d_real - d_vector;
  transformed_data_row_vector <- d_row_vector - d_real;
  transformed_data_row_vector <- d_real - d_row_vector;
  transformed_data_matrix <- d_matrix - d_real;
  transformed_data_matrix <- d_real - d_matrix;

  //   elementwise products
  transformed_data_vector <- d_vector .* d_vector;
  transformed_data_row_vector <- d_row_vector .* d_row_vector;
  transformed_data_matrix <- d_matrix .* d_matrix;
  transformed_data_vector <- d_vector ./ d_vector;
  transformed_data_row_vector <- d_row_vector ./ d_row_vector;
  transformed_data_matrix <- d_matrix ./ d_matrix;

  //   elementwise logarithms
  transformed_data_vector <- log(d_vector);
  transformed_data_row_vector <- log(d_row_vector);
  transformed_data_matrix <- log(d_matrix);
  transformed_data_vector <- exp(d_vector);
  transformed_data_row_vector <- exp(d_row_vector);
  transformed_data_matrix <- exp(d_matrix);

  //  dot products
  transformed_data_real <- dot_product(d_vector, d_vector);
  transformed_data_real <- dot_product(d_vector, d_row_vector);
  transformed_data_real <- dot_product(d_row_vector, d_vector);
  transformed_data_real <- dot_product(d_row_vector, d_row_vector);

  transformed_data_real <- dot_self(d_vector);
  transformed_data_real <- dot_self(d_row_vector);

  transformed_data_row_vector <- columns_dot_product(d_vector, d_vector);
  transformed_data_row_vector <- columns_dot_product(d_row_vector, d_row_vector);
  transformed_data_row_vector <- columns_dot_product(d_matrix, d_matrix);

  transformed_data_vector <- rows_dot_product(d_vector, d_vector);
  transformed_data_vector <- rows_dot_product(d_row_vector, d_row_vector);
  transformed_data_vector <- rows_dot_product(d_matrix, d_matrix);

  transformed_data_row_vector <- columns_dot_self(d_vector);
  transformed_data_row_vector <- columns_dot_self(d_row_vector);
  transformed_data_row_vector <- columns_dot_self(d_matrix);

  transformed_data_vector <- rows_dot_self(d_vector);
  transformed_data_vector <- rows_dot_self(d_row_vector);
  transformed_data_vector <- rows_dot_self(d_matrix);

  // quadratic forms
  transformed_data_real <- quad_form(d_matrix,d_vector);
  transformed_data_matrix <- quad_form(d_matrix,d_matrix);
  transformed_data_real <- trace_quad_form(d_matrix,d_vector);
  transformed_data_real <- trace_quad_form(d_matrix,d_matrix);
  transformed_data_real <- trace_gen_quad_form(d_matrix,d_matrix,d_matrix);

  //  reductions
  transformed_data_real <- min(d_vector);
  transformed_data_real <- min(d_row_vector);
  transformed_data_real <- min(d_matrix);
  transformed_data_real <- max(d_vector);
  transformed_data_real <- max(d_row_vector);
  transformed_data_real <- max(d_matrix);

  //  sums and products
  transformed_data_real <- sum(d_vector);
  transformed_data_real <- sum(d_row_vector);
  transformed_data_real <- sum(d_matrix);
  transformed_data_real <- prod(d_vector);
  transformed_data_real <- prod(d_row_vector);
  transformed_data_real <- prod(d_matrix);

  //  sample moments
  transformed_data_real <- mean(d_vector);
  transformed_data_real <- mean(d_row_vector);
  transformed_data_real <- mean(d_matrix);
  transformed_data_real <- variance(d_vector);
  transformed_data_real <- variance(d_row_vector);
  transformed_data_real <- variance(d_matrix);
  transformed_data_real <- sd(d_vector);
  transformed_data_real <- sd(d_row_vector);
  transformed_data_real <- sd(d_matrix);
  
  //Broadcast Functions
  transformed_data_vector <- rep_vector(d_real, d_int);
  transformed_data_row_vector <- rep_row_vector(d_real, d_int);
  transformed_data_matrix <- rep_matrix(d_real, d_int, d_int);
  transformed_data_matrix <- rep_matrix(d_vector, d_int);
  transformed_data_matrix <- rep_matrix(d_row_vector, d_int);
  transformed_data_vector <- to_vector(d_row_vector);
  transformed_data_vector <- to_vector(d_matrix);

  // Slice and Package Functions
  //   diagonal matrices
  transformed_data_vector <- diagonal(d_matrix);
  transformed_data_matrix <- diag_matrix(d_vector);
  transformed_data_vector <- col(d_matrix, d_int);
  transformed_data_row_vector <- row(d_matrix, d_int);
  
  //   transposition postfix operator
  transformed_data_matrix <- d_matrix';
  transformed_data_row_vector <- d_vector';
  transformed_data_vector <- d_row_vector';

  // Special Matrix Functions
  transformed_data_vector <- softmax(d_vector);

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
}
parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;
  real transformed_param_real_array[d_int];
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

  // logical functions
  transformed_param_real <- if_else(d_int, d_real, d_real);
  transformed_param_real <- if_else(d_int, p_real, d_real);
  transformed_param_real <- if_else(d_int, d_real, p_real);
  transformed_param_real <- if_else(d_int, p_real, p_real);
  transformed_param_real <- step(d_real);
  transformed_param_real <- step(p_real);
  
  // real-valued arithmetic operators
  //   binary infix operators
  transformed_param_real <- d_real + d_real;
  transformed_param_real <- p_real + d_real;
  transformed_param_real <- d_real + p_real;
  transformed_param_real <- p_real + p_real;
  transformed_param_real <- d_real - d_real;
  transformed_param_real <- p_real - d_real;
  transformed_param_real <- d_real - p_real;
  transformed_param_real <- p_real - p_real;
  transformed_param_real <- d_real * d_real;
  transformed_param_real <- p_real * d_real;
  transformed_param_real <- d_real * p_real;
  transformed_param_real <- p_real * p_real;
  transformed_param_real <- d_real / d_real;
  transformed_param_real <- p_real / d_real;
  transformed_param_real <- d_real / p_real;
  transformed_param_real <- p_real / p_real;

  //   unary prefix operators
  transformed_param_real <- -d_real;
  transformed_param_real <- -p_real;
  transformed_param_real <- +d_real;
  transformed_param_real <- +p_real;

  //   absolute functions
  transformed_param_real <- abs(d_real);
  transformed_param_real <- abs(p_real);
  transformed_param_real <- fabs(d_real);
  transformed_param_real <- fabs(p_real);
  transformed_param_real <- fdim(d_real, d_real);
  transformed_param_real <- fdim(p_real, d_real);
  transformed_param_real <- fdim(d_real, p_real);
  transformed_param_real <- fdim(p_real, p_real);

  //   bounds functions
  transformed_param_real <- fmin(d_real, d_real);
  transformed_param_real <- fmin(p_real, d_real);
  transformed_param_real <- fmin(d_real, p_real);
  transformed_param_real <- fmin(p_real, p_real);
  transformed_param_real <- fmax(d_real, d_real);
  transformed_param_real <- fmax(p_real, d_real);
  transformed_param_real <- fmax(d_real, p_real);
  transformed_param_real <- fmax(p_real, p_real);

  //   arithmetic functions
  transformed_param_real <- fmod(d_real, d_real);
  transformed_param_real <- fmod(p_real, d_real);
  transformed_param_real <- fmod(d_real, p_real);
  transformed_param_real <- fmod(p_real, p_real);

  //   rounding functions
  transformed_param_real <- floor(d_real);
  transformed_param_real <- floor(p_real);
  transformed_param_real <- ceil(d_real);
  transformed_param_real <- ceil(p_real);
  transformed_param_real <- round(d_real);
  transformed_param_real <- round(p_real);
  transformed_param_real <- trunc(d_real);
  transformed_param_real <- trunc(p_real);
  
  //   power and logarithm functions
  transformed_param_real <- sqrt(d_real);
  transformed_param_real <- sqrt(p_real);
  transformed_param_real <- cbrt(d_real);
  transformed_param_real <- cbrt(p_real);
  transformed_param_real <- square(d_real);
  transformed_param_real <- square(p_real);
  transformed_param_real <- exp(d_real);
  transformed_param_real <- exp(p_real);
  transformed_param_real <- exp2(d_real);
  transformed_param_real <- exp2(p_real);
  transformed_param_real <- log(d_real);
  transformed_param_real <- log(p_real);
  transformed_param_real <- log2(d_real);
  transformed_param_real <- log2(p_real);
  transformed_param_real <- log10(d_real);
  transformed_param_real <- log10(p_real);
  transformed_param_real <- pow(d_real, d_real);
  transformed_param_real <- pow(p_real, d_real);
  transformed_param_real <- pow(d_real, p_real);
  transformed_param_real <- pow(p_real, p_real);
  transformed_param_real <- inv(d_real);
  transformed_param_real <- inv(p_real);
  transformed_param_real <- inv_square(d_real);
  transformed_param_real <- inv_square(p_real);
  transformed_param_real <- inv_sqrt(d_real);
  transformed_param_real <- inv_sqrt(p_real);


  //   trigonometric functions
  transformed_param_real <- hypot(d_real, d_real);
  transformed_param_real <- hypot(p_real, d_real);
  transformed_param_real <- hypot(d_real, p_real);
  transformed_param_real <- hypot(p_real, p_real);
  transformed_param_real <- cos(d_real);
  transformed_param_real <- cos(p_real);
  transformed_param_real <- sin(d_real);
  transformed_param_real <- sin(p_real);
  transformed_param_real <- tan(d_real);
  transformed_param_real <- tan(p_real);
  transformed_param_real <- acos(d_real);
  transformed_param_real <- acos(p_real);
  transformed_param_real <- asin(d_real);
  transformed_param_real <- asin(p_real);
  transformed_param_real <- atan(d_real);
  transformed_param_real <- atan(p_real);
  transformed_param_real <- atan2(d_real, d_real);
  transformed_param_real <- atan2(p_real, d_real);
  transformed_param_real <- atan2(d_real, p_real);
  transformed_param_real <- atan2(p_real, p_real);

  //   hyperbolic trigonometric functions
  transformed_param_real <- cosh(d_real);
  transformed_param_real <- cosh(p_real);
  transformed_param_real <- sinh(d_real);
  transformed_param_real <- sinh(p_real);
  transformed_param_real <- tanh(d_real);
  transformed_param_real <- tanh(p_real);
  transformed_param_real <- acosh(d_real);
  transformed_param_real <- acosh(p_real);
  transformed_param_real <- asinh(d_real);
  transformed_param_real <- asinh(p_real);
  transformed_param_real <- atanh(d_real);
  transformed_param_real <- atanh(p_real);

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
  transformed_param_real <- Phi(d_real);
  transformed_param_real <- Phi(p_real);
  transformed_param_real <- Phi_approx(d_real);
  transformed_param_real <- Phi_approx(p_real);
  transformed_param_real <- binary_log_loss(d_int, d_real);
  transformed_param_real <- binary_log_loss(d_int, p_real);
  transformed_param_real <- owens_t(d_real, d_real);
  transformed_param_real <- owens_t(d_real, p_real);
  transformed_param_real <- owens_t(p_real, d_real);
  transformed_param_real <- owens_t(p_real, p_real);

  //   combinatorial functions
  transformed_param_real <- tgamma(d_real);
  transformed_param_real <- tgamma(p_real);
  transformed_param_real <- lgamma(d_real);
  transformed_param_real <- lgamma(p_real);
  transformed_param_real <- digamma(d_real);
  transformed_param_real <- digamma(p_real);
  transformed_param_real <- trigamma(d_real);
  transformed_param_real <- trigamma(p_real);
  transformed_param_real <- gamma_p(d_real, d_real);
  transformed_param_real <- gamma_p(p_real, d_real);
  transformed_param_real <- gamma_p(d_real, p_real);
  transformed_param_real <- gamma_p(p_real, p_real);
  transformed_param_real <- gamma_q(d_real, d_real);
  transformed_param_real <- gamma_q(p_real, d_real);
  transformed_param_real <- gamma_q(d_real, p_real);
  transformed_param_real <- gamma_q(p_real, p_real);
  transformed_param_real <- lmgamma(d_int, d_real);
  transformed_param_real <- lmgamma(d_int, p_real);
  transformed_param_real <- lbeta(d_real, d_real);
  transformed_param_real <- lbeta(p_real, d_real);
  transformed_param_real <- lbeta(d_real, p_real);
  transformed_param_real <- lbeta(p_real, p_real);
  transformed_param_real <- binomial_coefficient_log(d_real, d_real);
  transformed_param_real <- binomial_coefficient_log(p_real, d_real);
  transformed_param_real <- binomial_coefficient_log(d_real, p_real);
  transformed_param_real <- binomial_coefficient_log(p_real, p_real);
  transformed_param_real <- bessel_first_kind(d_int, d_real);
  transformed_param_real <- bessel_first_kind(d_int, p_real);
  transformed_param_real <- bessel_second_kind(d_int, d_real);
  transformed_param_real <- bessel_second_kind(d_int, p_real);
  transformed_param_real <- modified_bessel_first_kind(d_int, d_real);
  transformed_param_real <- modified_bessel_first_kind(d_int, p_real);
  transformed_param_real <- modified_bessel_second_kind(d_int, d_real);
  transformed_param_real <- modified_bessel_second_kind(d_int, p_real);
  transformed_param_real <- falling_factorial(d_real, d_real);
  transformed_param_real <- falling_factorial(p_real, d_real);  
  transformed_param_real <- falling_factorial(d_real, p_real);  
  transformed_param_real <- falling_factorial(p_real, p_real);  
  transformed_param_real <- rising_factorial(d_real, d_real);
  transformed_param_real <- rising_factorial(p_real, d_real);
  transformed_param_real <- rising_factorial(d_real, p_real);
  transformed_param_real <- rising_factorial(p_real, p_real);
  transformed_param_real <- log_falling_factorial(d_real, d_real);
  transformed_param_real <- log_falling_factorial(p_real, d_real); 
  transformed_param_real <- log_falling_factorial(d_real, p_real); 
  transformed_param_real <- log_falling_factorial(p_real, p_real); 
  transformed_param_real <- log_rising_factorial(d_real, d_real);
  transformed_param_real <- log_rising_factorial(p_real, d_real);  
  transformed_param_real <- log_rising_factorial(d_real, p_real);  
  transformed_param_real <- log_rising_factorial(p_real, p_real);  


  //   composed functions
  transformed_param_real <- expm1(d_real);
  transformed_param_real <- expm1(p_real);
  transformed_param_real <- fma(d_real, d_real, d_real);
  transformed_param_real <- fma(d_real, d_real, p_real);
  transformed_param_real <- fma(d_real, p_real, d_real);
  transformed_param_real <- fma(d_real, p_real, p_real);
  transformed_param_real <- fma(p_real, d_real, d_real);
  transformed_param_real <- fma(p_real, d_real, p_real);
  transformed_param_real <- fma(p_real, p_real, d_real);
  transformed_param_real <- fma(p_real, p_real, p_real);
  transformed_param_real <- multiply_log(d_real, d_real);
  transformed_param_real <- multiply_log(p_real, d_real);
  transformed_param_real <- multiply_log(d_real, p_real);
  transformed_param_real <- multiply_log(p_real, p_real);
  transformed_param_real <- log1p(d_real);
  transformed_param_real <- log1p(p_real);
  transformed_param_real <- log1m(d_real);
  transformed_param_real <- log1m(p_real);
  transformed_param_real <- log1p_exp(d_real);
  transformed_param_real <- log1p_exp(p_real);
  transformed_param_real <- log_inv_logit(d_real);
  transformed_param_real <- log_inv_logit(p_real);
  transformed_param_real <- log1m_inv_logit(d_real);
  transformed_param_real <- log1m_inv_logit(p_real);
  transformed_param_real <- log_sum_exp(d_real, d_real);
  transformed_param_real <- log_sum_exp(p_real, d_real);
  transformed_param_real <- log_sum_exp(d_real, p_real);
  transformed_param_real <- log_sum_exp(p_real, p_real);

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
  transformed_param_real <- int_step(rows(d_vector)); // using int_step to test integer output
  transformed_param_real <- int_step(rows(p_vector)); 
  transformed_param_real <- int_step(rows(d_row_vector));
  transformed_param_real <- int_step(rows(p_row_vector));
  transformed_param_real <- int_step(rows(d_matrix));
  transformed_param_real <- int_step(rows(p_matrix));
  transformed_param_real <- int_step(cols(d_vector)); // using int_step to test integer output
  transformed_param_real <- int_step(cols(p_vector)); 
  transformed_param_real <- int_step(cols(d_row_vector));
  transformed_param_real <- int_step(cols(p_row_vector));
  transformed_param_real <- int_step(cols(d_matrix));
  transformed_param_real <- int_step(cols(p_matrix));

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

  transformed_param_vector <- to_vector(d_row_vector);
  transformed_param_vector <- to_vector(p_row_vector);
  transformed_param_vector <- to_vector(d_matrix);
  transformed_param_vector <- to_vector(p_matrix);

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
}
model {  
}
