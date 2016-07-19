data {
  int a0;
  real b0;
}

transformed data {
  int td_a = a0;
  int td_a0 = 3;
  real td_b = b0;
  real td_b0 = 4.0;
  {
    int loc_td_a = a0;
    int loc_td_a0 = 5;
    real loc_td_b = b0;
    real loc_td_b0 = 6.0;
  }
}

parameters {
  real par_b = b0;
  real par_b0 = 0.0;
}

transformed parameters {
  real tpar_b = b0;
  real tpar_b0 = 1.0
  {
    int loc_tpar_a = a0;
    int loc_tpar_a0 = 2.0;
  }
}

model {
  real model_b = b0;
  {
    int loc_model_a = a0;
    int loc_model_a0 = 3;
    real loc_model_b = b0;
    real loc_model_b0 = 4.0;
  }

}

generated quantities {
  real gq_b = b0;
  real gq_b0 = 9.99;
  {
    int loc_gq_a = a0;
    int loc_gq_a0 = 7;
    real loc_gq_b = b0;
    real loc_gq_b0 = 6.66;
  }
}
