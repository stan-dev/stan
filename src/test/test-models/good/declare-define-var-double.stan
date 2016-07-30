data {
  real b0;
}
transformed data {
  real td_a;
  real td_b = b0;
  real td_b0 = 4.0;
  {
    real loc_td_a;
    real loc_td_b = b0;
    real loc_td_b0 = 6.0;
  }
}
parameters {
  real par_a;
  real par_b = b0;
  real par_b0 = 0.0;
}
transformed parameters {
  real tpar_a;
  real tpar_b = b0;
  real tpar_b0 = 1.0;
  {
    real loc_tpar_a0;
    real loc_tpar_b0 = 2.0;
  }
}
model {
  real model_a;
  real model_b = b0;
  {
    real loc_model_a;
    real loc_model_b = b0;
    real loc_model_b0 = 4.0;
  }
}
generated quantities {
  real gq_a;
  real gq_b = b0;
  real gq_b0 = 9.99;
  {
    real loc_gq_a;
    real loc_gq_b = b0;
    real loc_gq_b0 = 6.66;
  }
}
