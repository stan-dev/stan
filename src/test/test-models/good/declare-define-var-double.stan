transformed data {
  real td1 = 1;     // real_d <- int_d
  real td2 = 2.0;   // real_d <- real_d
  real td3[7];
  real td4[7] = td3;
}
transformed parameters {
  // block variables
  real p1 = 1;       // real_p <- int_d 
  real p2 = 2.0;     // real_p <- real_d
  real p3 = p2;      // real_p <- real_p
  p3 = p2;
  // local variables
  {
    real lp1 = 1;      // real_p <- int_d 
    real lp2 = 2.0;    // real_p <- real_d
    real lp3 = p2;      // real_p <- real_p
    p3 = p2;
  }
}
model {
  // local variables
  real lm1 = 1;      // real_p <- int_d 
  real lm2 = 2.0;    // real_p <- real_d
  real lm3 = p2;      // real_p <- real_p
}
generated quantities {
  real gq1 = 1;     // real_d <- int_d
  real gq2 = 2.0;   // real_d <- real_d
  gq2 = 5.0;
  {
    real lgq1 = 1;     // real_d <- int_d
    real lgq2 = 2.0;   // real_d <- real_d
    lgq2 = 5.0;
  }
}


