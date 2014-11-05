transformed data {
  real a[109,307];
  vector[5] b[17];
  row_vector[5] c[17];
  matrix[15,27] d;

  real aa[12,12,12];
  vector[5] bb[12,12];
  row_vector[5] cc[12,12];
  matrix[5,12] dd[12];

  a[1][1] <- 118.22;
  b[1][1] <- 13;
  c[1][1] <- 0;
  d[1][1] <- 12;

  aa[1][1][1] <- 118.22;
  bb[1][1][1] <- 13;
  cc[1][1][1] <- 0;
  dd[1][1][1] <- 12;
}
parameters {
  real y;
}
transformed parameters {
  real ap[109,307];
  vector[5] bp[17];
  row_vector[5] cp[17];
  matrix[15,27] dp;

  real aap[12,12,12];
  vector[5] bbp[12,12];
  row_vector[5] ccp[12,12];
  matrix[5,12] ddp[12];

  ap[1][1] <- 118.22;
  bp[1][1] <- 13;
  cp[1][1] <- 0;
  dp[1][1] <- 12;

  aap[1][1][1] <- 118.22;
  bbp[1][1][1] <- 13;
  ccp[1][1][1] <- 0;
  ddp[1][1][1] <- 12;
}
model {
  y ~ normal(0,1);
}
