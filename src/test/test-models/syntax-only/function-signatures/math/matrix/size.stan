data {
  int x1x;
  real x2x;
  vector[2] x3x;
  row_vector[2] x4x;
  matrix[2,3] x5x;

  int x1y[3];
  real x2y[3];
  vector[2] x3y[3];
  row_vector[2] x4y[3];
  matrix[2,3] x5y[3];

  int x1z[3,4];
  real x2z[3,4];
  vector[2] x3z[3,4];
  row_vector[2] x4z[3,4];
  matrix[2,3] x5z[3,4];

  int x1w[3,4,5];
  real x2w[3,4,5];
  vector[2] x3w[3,4,5];
  row_vector[2] x4w[3,4,5];
  matrix[2,3] x5w[3,4,5];
}
transformed data {
  int val;

  val <- size(x1y);
  val <- size(x2y);
  val <- size(x3y);
  val <- size(x4y);
  val <- size(x5y);

  val <- size(x1z);
  val <- size(x2z);
  val <- size(x3z);
  val <- size(x4z);
  val <- size(x5z);

  val <- size(x1w);
  val <- size(x2w);
  val <- size(x3w);
  val <- size(x4w);
  val <- size(x5w);
}
parameters {
  real y;

  real p_x1x;
  real p_x2x;
  vector[2] p_x3x;
  row_vector[2] p_x4x;
  matrix[2,3] p_x5x;

  real p_x1y[3];
  real p_x2y[3];
  vector[2] p_x3y[3];
  row_vector[2] p_x4y[3];
  matrix[2,3] p_x5y[3];

  real p_x1z[3,4];
  real p_x2z[3,4];
  vector[2] p_x3z[3,4];
  row_vector[2] p_x4z[3,4];
  matrix[2,3] p_x5z[3,4];

  real p_x1w[3,4,5];
  real p_x2w[3,4,5];
  vector[2] p_x3w[3,4,5];
  row_vector[2] p_x4w[3,4,5];
  matrix[2,3] p_x5w[3,4,5];
}
transformed parameters {
  real p_val;

  p_val <- size(x1y);
  p_val <- size(x2y);
  p_val <- size(x3y);
  p_val <- size(x4y);
  p_val <- size(x5y);

  p_val <- size(x1z);
  p_val <- size(x2z);
  p_val <- size(x3z);
  p_val <- size(x4z);
  p_val <- size(x5z);

  p_val <- size(x1w);
  p_val <- size(x2w);
  p_val <- size(x3w);
  p_val <- size(x4w);
  p_val <- size(x5w);

  p_val <- size(p_x1y);
  p_val <- size(p_x2y);
  p_val <- size(p_x3y);
  p_val <- size(p_x4y);
  p_val <- size(p_x5y);

  p_val <- size(p_x1z);
  p_val <- size(p_x2z);
  p_val <- size(p_x3z);
  p_val <- size(p_x4z);
  p_val <- size(p_x5z);

  p_val <- size(p_x1w);
  p_val <- size(p_x2w);
  p_val <- size(p_x3w);
  p_val <- size(p_x4w);
  p_val <- size(p_x5w);
}
model {
  y ~ normal(0,1);
}
