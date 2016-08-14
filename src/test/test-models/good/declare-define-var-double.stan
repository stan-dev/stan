data {
  real b0;
}
transformed data {
  real<lower=0,upper=1> td_b1;
  real<lower=1,upper=2> td_b2 = b0;
  real<lower=0> td_b3 = 4;
  real<upper=1> td_b4 = 4.4;
  {
    real loc_td_a;
    real loc_td_b1 = b0;
    real loc_td_b2 = 6.6;
  }
}
parameters {
  real par_a;
  real par_b1 = b0;
  real par_b2 = 2.3;
}
transformed parameters {
  real tpar_a;
  real tpar_b1 = b0;
  real tpar_b2 = 1.1;
  {
    real loc_tpar_a0;
    real loc_tpar_b1 = b0;
    real loc_tpar_b2 = 6.6;
  }
}
model {
  real model_a;
  real model_b1 = b0;
  real model_b2 = 3.3;
  {
    real loc_model_a;
    real loc_model_b1 = b0;
    real loc_model_b2 = 4.4;
  }
}
generated quantities {
  real gq_a;
  real gq_b1 = b0;
  real gq_b2 = 9.99;
  {
    real loc_gq_a;
    real loc_gq_b1 = b0;
    real loc_gq_b2 = 6.66;
  }
}
