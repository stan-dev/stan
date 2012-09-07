data { 
  int d_int;
  
  real d_real;
}
transformed data{
  int transformed_data_int;
  real transformed_data_real;

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
}
parameters {
  real p_real;
}
transformed parameters {
  real transformed_param_real;

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
}
model {
}
