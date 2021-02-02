transformed data {
  array[109, 307] real a;
  array[17] vector[5] b;
  array[17] row_vector[5] c;
  matrix[15, 27] d;
  array[12, 12, 12] real aa;
  array[12, 12] vector[5] bb;
  array[12, 12] row_vector[5] cc;
  array[12] matrix[5, 12] dd;
  a[1][1] = 118.22;
  b[1][1] = 13;
  c[1][1] = 0;
  d[1][1] = 12;
  aa[1][1][1] = 118.22;
  bb[1][1][1] = 13;
  cc[1][1][1] = 0;
  dd[1][1][1] = 12;
}
parameters {
  real y;
}
transformed parameters {
  array[109, 307] real ap;
  array[17] vector[5] bp;
  array[17] row_vector[5] cp;
  matrix[15, 27] dp;
  array[12, 12, 12] real aap;
  array[12, 12] vector[5] bbp;
  array[12, 12] row_vector[5] ccp;
  array[12] matrix[5, 12] ddp;
  ap[1][1] = 118.22;
  bp[1][1] = 13;
  cp[1][1] = 0;
  dp[1][1] = 12;
  aap[1][1][1] = 118.22;
  bbp[1][1][1] = 13;
  ccp[1][1][1] = 0;
  ddp[1][1][1] = 12;
}
model {
  y ~ normal(0, 1);
}

