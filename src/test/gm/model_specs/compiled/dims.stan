transformed data {
  int val;
  int val0[0];
  int val1[1];
  int val2[2];
  int val3[3];
  int val4[4];
  int val5[5];
  int val6[6];
  int val7[7];
  int val8[8];
  int val9[9];
  int val10[10];

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

  val0 <- dims(x1x);
  val0 <- dims(x2x);
  val1 <- dims(x3x);
  val1 <- dims(x4x);
  val2 <- dims(x5x);

  val1 <- dims(x1y);
  val1 <- dims(x2y);
  val2 <- dims(x3y);
  val2 <- dims(x4y);
  val3 <- dims(x5y);

  val2 <- dims(x1z);
  val2 <- dims(x2z);
  val3 <- dims(x3z);
  val3 <- dims(x4z);
  val4 <- dims(x5z);

  val3 <- dims(x1w);
  val3 <- dims(x2w);
  val4 <- dims(x3w);
  val4 <- dims(x4w);
  val5 <- dims(x5w);

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
}
model {
  y ~ normal(0,1);
}
