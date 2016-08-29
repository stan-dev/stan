functions {
  real foo() {
    real lf1 = 1;     // real_d <- int_d
    real lf2 = 2.0;   // real_d <- real_d
    return lf1;
  }
}
transformed data {
  real td1 = 1;     // real_d <- int_d
  real td2 = 2.0;   // real_d <- real_d

  real td_a1[5];
  real td_a2[5] = td_a1;  // real_arr_d <- real_arr_d

  // local variables
  {
    real ltd1 = 1;      // real_d <- int_d 
    real ltd2 = 2.0;    // real_d <- real_d
  }
}
transformed parameters {
  // block variables
  real tp1 = 1;       // real_p <- int_d 
  real tp2 = 2.0;     // real_p <- real_d
  real tp3 = tp2;      // real_p <- real_p
  // local variables
  {
    real lp1 = 1;      // real_p <- int_d 
    real lp2 = 2.0;    // real_p <- real_d
    real lp3 = tp2;      // real_p <- real_p
    tp3 = tp2;
  }
}
model {
  // local variables
  real lm1 = 1;      // real_p <- int_d 
  real lm2 = 2.0;    // real_p <- real_d
  real lm3 = tp2;      // real_p <- real_p
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
