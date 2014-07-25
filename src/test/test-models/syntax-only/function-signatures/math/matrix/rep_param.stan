parameters {
  real zzz;
}
transformed parameters {
  real x0;
  real x1[2];
  real x2[3,2];
  real x3[4,3,2];

  // no int params yet
  // int n0;
  // int n1[2];
  // int n2[3,2];
  // int n3[4,3,2];
  
  vector[5] v0;
  vector[5] v1[6];
  vector[5] v2[7,6];
  vector[5] v3[8,7,6];

  row_vector[5] rv0;
  row_vector[5] rv1[6];
  row_vector[5] rv2[7,6];
  row_vector[5] rv3[8,7,6];

  matrix[5,6] m0;
  matrix[5,6] m1[9];
  matrix[5,6] m2[10,9];
  matrix[5,6] m3[11,10,9];

  x0 <- 10.9;
  // n0 <- 5;

  // rep_vector
  v0 <- rep_vector(3.7,5);

  // rep_row_vector
  rv0 <- rep_row_vector(9.8,5);

  // rep_matrix (2D)
  m0 <- rep_matrix(-1.8, 5, 6);

  // rep_array 1D
  // n1 <- rep_array(n0,2);
  // n2 <- rep_array(n1,3);
  // n3 <- rep_array(n2,4);

  x1 <- rep_array(x0,2);
  x2 <- rep_array(x1,3);
  x3 <- rep_array(x2,4);

  v1 <- rep_array(v0,6);
  v2 <- rep_array(v1,7);
  v3 <- rep_array(v2,8);

  rv1 <- rep_array(rv0,6);
  rv2 <- rep_array(rv1,7);
  rv3 <- rep_array(rv2,8);

  m1 <- rep_array(m0,6);
  m2 <- rep_array(m1,7);
  m3 <- rep_array(m2,8);

  // rep_array 2D
  x2 <- rep_array(-128.676, 3, 2);

  // rep_array 3D
  x3 <- rep_array(-128.676, 4, 3, 2);

}
model {
  zzz ~ normal(0,1);
}
