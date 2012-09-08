data { 
  int d_int;
  real d_real;
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data{
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
  transformed_data_real <- epsilon();
  transformed_data_real <- negative_epsilon();

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
  transformed_data_real <- binary_log_loss(d_int, d_real);

  //   combinatorial functions
  transformed_data_real <- tgamma(d_real);
  transformed_data_real <- lgamma(d_real);
  transformed_data_real <- lmgamma(d_int, d_real);
  transformed_data_real <- lbeta(d_real, d_real);
  transformed_data_real <- binomial_coefficient_log(d_real, d_real);

  //   composed functions
  transformed_data_real <- expm1(d_real);
  transformed_data_real <- fma(d_real, d_real, d_real);
  transformed_data_real <- multiply_log(d_real, d_real);
  transformed_data_real <- log1p(d_real);
  transformed_data_real <- log1m(d_real);
  transformed_data_real <- log1p_exp(d_real);
  transformed_data_real <- log_sum_exp(d_real, d_real);

  //*** Array Operations ***
  transformed_data_real <- sum(d_real_array);
  transformed_data_real <- mean(d_real_array);
  transformed_data_real <- variance(d_real_array);
  transformed_data_real <- sd(d_real_array);
  transformed_data_real <- log_sum_exp(d_real_array);
  
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
  transformed_param_real <- epsilon();
  transformed_param_real <- negative_epsilon();

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
  transformed_param_real <- binary_log_loss(d_int, d_real);
  transformed_param_real <- binary_log_loss(d_int, p_real);

  //   combinatorial functions
  transformed_param_real <- tgamma(d_real);
  transformed_param_real <- tgamma(p_real);
  transformed_param_real <- lgamma(d_real);
  transformed_param_real <- lgamma(p_real);
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
  //FIXME: transformed_param_vector <- -d_vector;
  transformed_param_vector <- -p_vector;
  //FIXME: transformed_param_row_vector <- -d_row_vector;
  transformed_param_row_vector <- -p_row_vector;
  //FIXME: transformed_param_matrix <- -d_matrix;
  transformed_param_matrix <- -p_matrix;

  //   infix matrix operators
  //FIXME: transformed_param_vector <- d_vector + d_vector;
  transformed_param_vector <- p_vector + d_vector;
  transformed_param_vector <- d_vector + p_vector;
  transformed_param_vector <- p_vector + p_vector;
  //FIXME: transformed_param_row_vector <- d_row_vector + d_row_vector;
  transformed_param_row_vector <- p_row_vector + d_row_vector;
  transformed_param_row_vector <- d_row_vector + p_row_vector;
  transformed_param_row_vector <- p_row_vector + p_row_vector;
  //FIXME: transformed_param_matrix <- d_matrix + d_matrix;
  transformed_param_matrix <- p_matrix + d_matrix;
  transformed_param_matrix <- d_matrix + p_matrix;
  transformed_param_matrix <- p_matrix + p_matrix;
  //FIXME: transformed_param_vector <- d_vector - d_vector;
  transformed_param_vector <- p_vector - d_vector;
  transformed_param_vector <- d_vector - p_vector;
  transformed_param_vector <- p_vector - p_vector;
  //FIXME: transformed_param_row_vector <- d_row_vector - d_row_vector;
  transformed_param_row_vector <- p_row_vector - d_row_vector;
  transformed_param_row_vector <- d_row_vector - p_row_vector;
  transformed_param_row_vector <- p_row_vector - p_row_vector;
  //FIXME: transformed_param_matrix <- d_matrix - d_matrix;
  transformed_param_matrix <- p_matrix - d_matrix;
  transformed_param_matrix <- d_matrix - p_matrix;
  transformed_param_matrix <- p_matrix - p_matrix;
  //FIXME: transformed_param_vector <- d_real * d_vector;
  transformed_param_vector <- p_real * d_vector;
  transformed_param_vector <- d_real * p_vector;
  transformed_param_vector <- p_real * p_vector;
  //FIXME: transformed_param_row_vector <- d_real * d_row_vector;
  transformed_param_row_vector <- p_real * d_row_vector;
  transformed_param_row_vector <- d_real * p_row_vector;
  transformed_param_row_vector <- p_real * p_row_vector;
  //FIXME: transformed_param_matrix <- d_real * d_matrix;
  transformed_param_matrix <- p_real * d_matrix;
  transformed_param_matrix <- d_real * p_matrix;
  transformed_param_matrix <- p_real * p_matrix;
  //FIXME: transformed_param_vector <- d_vector * d_real;
  transformed_param_vector <- p_vector * d_real;
  transformed_param_vector <- d_vector * p_real;
  transformed_param_vector <- p_vector * p_real;
  //FIXME: transformed_param_row_vector <- d_row_vector * d_real;
  transformed_param_row_vector <- p_row_vector * d_real;
  transformed_param_row_vector <- d_row_vector * p_real;
  transformed_param_row_vector <- p_row_vector * p_real;
  //FIXME: transformed_param_matrix <- d_matrix * d_real;
  transformed_param_matrix <- p_matrix * d_real;
  transformed_param_matrix <- d_matrix * p_real;
  transformed_param_matrix <- p_matrix * p_real;
  transformed_param_real <- d_row_vector * d_vector;
  transformed_param_real <- p_row_vector * d_vector;
  transformed_param_real <- d_row_vector * p_vector;
  transformed_param_real <- p_row_vector * p_vector;
  //FIXME: transformed_param_row_vector <- d_row_vector * d_matrix;
  transformed_param_row_vector <- p_row_vector * d_matrix;
  transformed_param_row_vector <- d_row_vector * p_matrix;
  transformed_param_row_vector <- p_row_vector * p_matrix;
  //FIXME: transformed_param_matrix <- d_matrix * d_real;
  transformed_param_matrix <- p_matrix * d_real;
  transformed_param_matrix <- d_matrix * p_real;
  transformed_param_matrix <- p_matrix * p_real;
  //FIXME: transformed_param_vector <- d_matrix * d_vector;
  transformed_param_vector <- p_matrix * d_vector;
  transformed_param_vector <- d_matrix * p_vector;
  transformed_param_vector <- p_matrix * p_vector;
  //FIXME: transformed_param_matrix <- d_matrix * d_matrix;
  transformed_param_matrix <- p_matrix * d_matrix;
  transformed_param_matrix <- d_matrix * p_matrix;
  transformed_param_matrix <- p_matrix * p_matrix;
}
model {
}
