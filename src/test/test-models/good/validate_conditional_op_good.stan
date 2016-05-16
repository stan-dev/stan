//transformed data:
//  int, int
//  int, real
//  real, int
//  real, real
//  vector, vector
//  row_vector, row_vector
//  matrix, matrix
//
//  + array types 1 & 2 dims

parameters {
  real y;
}

//transformed parameters or model:
//  int, real
//  real, int
//  real, real
//  vector, vector
//  row_vector, row_vector
//  matrix, matrix
//  
//  + array types 1 & 2 dims


model {
  int x;
  real z;
  x = 0 ? x : 2;
  z = 1 ? x : z;

  y ~ normal(0,1);
}

