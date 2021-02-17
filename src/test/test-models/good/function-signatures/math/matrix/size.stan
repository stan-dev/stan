data {
  int x1x;
  real x2x;
  vector[2] x3x;
  row_vector[2] x4x;
  matrix[2, 3] x5x;
  array[3] int x1y;
  array[3] real x2y;
  array[3] vector[2] x3y;
  array[3] row_vector[2] x4y;
  array[3] matrix[2, 3] x5y;
  array[3, 4] int x1z;
  array[3, 4] real x2z;
  array[3, 4] vector[2] x3z;
  array[3, 4] row_vector[2] x4z;
  array[3, 4] matrix[2, 3] x5z;
  array[3, 4, 5] int x1w;
  array[3, 4, 5] real x2w;
  array[3, 4, 5] vector[2] x3w;
  array[3, 4, 5] row_vector[2] x4w;
  array[3, 4, 5] matrix[2, 3] x5w;
}
transformed data {
  int val;
  val = size(x1y);
  val = size(x2y);
  val = size(x3y);
  val = size(x4y);
  val = size(x5y);
  val = size(x1z);
  val = size(x2z);
  val = size(x3z);
  val = size(x4z);
  val = size(x5z);
  val = size(x1w);
  val = size(x2w);
  val = size(x3w);
  val = size(x4w);
  val = size(x5w);
}
parameters {
  real y;
  real p_x1x;
  real p_x2x;
  vector[2] p_x3x;
  row_vector[2] p_x4x;
  matrix[2, 3] p_x5x;
  array[3] real p_x1y;
  array[3] real p_x2y;
  array[3] vector[2] p_x3y;
  array[3] row_vector[2] p_x4y;
  array[3] matrix[2, 3] p_x5y;
  array[3, 4] real p_x1z;
  array[3, 4] real p_x2z;
  array[3, 4] vector[2] p_x3z;
  array[3, 4] row_vector[2] p_x4z;
  array[3, 4] matrix[2, 3] p_x5z;
  array[3, 4, 5] real p_x1w;
  array[3, 4, 5] real p_x2w;
  array[3, 4, 5] vector[2] p_x3w;
  array[3, 4, 5] row_vector[2] p_x4w;
  array[3, 4, 5] matrix[2, 3] p_x5w;
}
transformed parameters {
  real p_val;
  p_val = size(x1y);
  p_val = size(x2y);
  p_val = size(x3y);
  p_val = size(x4y);
  p_val = size(x5y);
  p_val = size(x1z);
  p_val = size(x2z);
  p_val = size(x3z);
  p_val = size(x4z);
  p_val = size(x5z);
  p_val = size(x1w);
  p_val = size(x2w);
  p_val = size(x3w);
  p_val = size(x4w);
  p_val = size(x5w);
  p_val = size(p_x1y);
  p_val = size(p_x2y);
  p_val = size(p_x3y);
  p_val = size(p_x4y);
  p_val = size(p_x5y);
  p_val = size(p_x1z);
  p_val = size(p_x2z);
  p_val = size(p_x3z);
  p_val = size(p_x4z);
  p_val = size(p_x5z);
  p_val = size(p_x1w);
  p_val = size(p_x2w);
  p_val = size(p_x3w);
  p_val = size(p_x4w);
  p_val = size(p_x5w);
}
model {
  y ~ normal(0, 1);
}

