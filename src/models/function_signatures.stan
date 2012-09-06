data { 
  int d_int_1;
  int d_int_2;
  int d_int_3;
  
  real d_real_1;
  real d_real_2;
  real d_real_3;
}
transformed data{
  int transformed_data_int;
  real transformed_data_real;

  //*** Integer-Valued Basic Functions ***
  // integer-valued arithmetic operators
  //   binary infix operators
  transformed_data_int <- d_int_1 + d_int_2;
  transformed_data_int <- d_int_1 - d_int_2;
  transformed_data_int <- d_int_1 * d_int_2;
  transformed_data_int <- d_int_1 / d_int_2;

  //   unary prefix operators
  transformed_data_int <- -d_int_1;
  transformed_data_int <- +d_int_1;
  
  //   absolute functions
  transformed_data_int <- abs(d_int_1);
  transformed_data_int <- int_step(d_int_1);

  //   bound functions
  transformed_data_int <- min(d_int_1, d_int_2);
  transformed_data_int <- max(d_int_1, d_int_2);

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
  transformed_data_real <- if_else(d_int_1, d_real_1, d_real_2);
  transformed_data_real <- step(d_real_1);
  
  // real-valued arithmetic operators
  //   binary infix operators
  transformed_data_real <- d_real_1 + d_real_2;
  transformed_data_real <- d_real_1 - d_real_2;
  transformed_data_real <- d_real_1 * d_real_2;
  transformed_data_real <- d_real_1 / d_real_2;

  //   unary prefix operators
  transformed_data_real <- -d_real_1;
  transformed_data_real <- +d_real_1;

  //   absolute functions
  transformed_data_real <- abs(d_real_1);
  transformed_data_real <- fabs(d_real_1);
  transformed_data_real <- fdim(d_real_1, d_real_2);

  //   bounds functions
  transformed_data_real <- fmin(d_real_1, d_real_2);
  transformed_data_real <- fmax(d_real_1, d_real_2);

  //   arithmetic functions
  transformed_data_real <- fmod(d_real_1, d_real_2);

  //   rounding functions
  transformed_data_real <- floor(d_real_1);
  transformed_data_real <- ceil(d_real_1);
  transformed_data_real <- round(d_real_1);
  transformed_data_real <- trunc(d_real_1);
}
parameters {
  real p_real_1;
  real p_real_2;
  real p_real_3;
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
  transformed_param_real <- if_else(d_int_1, d_real_1, d_real_2);
  transformed_param_real <- if_else(d_int_1, p_real_1, d_real_2);
  transformed_param_real <- if_else(d_int_1, d_real_1, p_real_2);
  transformed_param_real <- if_else(d_int_1, p_real_1, p_real_2);
  transformed_param_real <- step(d_real_1);
  transformed_param_real <- step(p_real_1);
  
  // real-valued arithmetic operators
  //   binary infix operators
  transformed_param_real <- d_real_1 + d_real_2;
  transformed_param_real <- p_real_1 + d_real_2;
  transformed_param_real <- d_real_1 + p_real_2;
  transformed_param_real <- p_real_1 + p_real_2;
  transformed_param_real <- d_real_1 - d_real_2;
  transformed_param_real <- p_real_1 - d_real_2;
  transformed_param_real <- d_real_1 - p_real_2;
  transformed_param_real <- p_real_1 - p_real_2;
  transformed_param_real <- d_real_1 * d_real_2;
  transformed_param_real <- p_real_1 * d_real_2;
  transformed_param_real <- d_real_1 * p_real_2;
  transformed_param_real <- p_real_1 * p_real_2;
  transformed_param_real <- d_real_1 / d_real_2;
  transformed_param_real <- p_real_1 / d_real_2;
  transformed_param_real <- d_real_1 / p_real_2;
  transformed_param_real <- p_real_1 / p_real_2;

  //   unary prefix operators
  transformed_param_real <- -d_real_1;
  transformed_param_real <- -p_real_1;
  transformed_param_real <- +d_real_1;
  transformed_param_real <- +p_real_1;

  //   absolute functions
  transformed_param_real <- abs(d_real_1);
  transformed_param_real <- abs(p_real_1);
  transformed_param_real <- fabs(d_real_1);
  transformed_param_real <- fabs(p_real_1);
  transformed_param_real <- fdim(d_real_1, d_real_2);
  transformed_param_real <- fdim(p_real_1, d_real_2);
  transformed_param_real <- fdim(d_real_1, p_real_2);
  transformed_param_real <- fdim(p_real_1, p_real_2);

  //   bounds functions
  transformed_param_real <- fmin(d_real_1, d_real_2);
  transformed_param_real <- fmin(p_real_1, d_real_2);
  transformed_param_real <- fmin(d_real_1, p_real_2);
  transformed_param_real <- fmin(p_real_1, p_real_2);
  transformed_param_real <- fmax(d_real_1, d_real_2);
  transformed_param_real <- fmax(p_real_1, d_real_2);
  transformed_param_real <- fmax(d_real_1, p_real_2);
  transformed_param_real <- fmax(p_real_1, p_real_2);

  //   arithmetic functions
  transformed_param_real <- fmod(d_real_1, d_real_2);
  transformed_param_real <- fmod(p_real_1, d_real_2);
  transformed_param_real <- fmod(d_real_1, p_real_2);
  transformed_param_real <- fmod(p_real_1, p_real_2);

  //   rounding functions
  transformed_param_real <- floor(d_real_1);
  transformed_param_real <- floor(p_real_1);
  transformed_param_real <- ceil(d_real_1);
  transformed_param_real <- ceil(p_real_1);
  transformed_param_real <- round(d_real_1);
  transformed_param_real <- round(p_real_1);
  transformed_param_real <- trunc(d_real_1);
  transformed_param_real <- trunc(p_real_1);
  

/*
  transformed_param_real <- d_real_1;
  transformed_param_real <- d_real_1;
  transformed_param_real <- d_real_1;
  transformed_param_real <- d_real_1;
  transformed_param_real <- d_real_1;
  transformed_param_real <- d_real_1;

*/

}
model {
}
