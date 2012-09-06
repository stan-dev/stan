data { 
  int data_int_1;
  int data_int_2;
  int data_int_3;
  
  real data_real_1;
  real data_real_2;
  real data_real_3;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

  //*** Integer-Valued Basic Functions ***
  // binary infix operators
  transformed_data_int <- data_int_1 + data_int_2;
  transformed_data_int <- data_int_1 - data_int_2;
  transformed_data_int <- data_int_1 * data_int_2;
  transformed_data_int <- data_int_1 / data_int_2;

  // unary prefix operators
  transformed_data_int <- -data_int_1;
  transformed_data_int <- +data_int_1;
  
  // absolute functions
  transformed_data_int <- abs(data_int_1);
  transformed_data_int <- int_step(data_int_1);

  // bound functions
  transformed_data_int <- min(data_int_1, data_int_2);
  transformed_data_int <- max(data_int_1, data_int_2);

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
  transformed_data_real <- if_else(data_int_1, data_real_1, data_real_2);
  transformed_data_real <- step(data_real_1);

/*
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;
  transformed_data_real <- data_real_1;

*/

}
parameters {
} 
transformed parameters {
}
model {
}
