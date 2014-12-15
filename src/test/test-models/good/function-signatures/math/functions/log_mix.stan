data { 
  real d_real_theta;
  real d_real_lam_1;
  real d_real_lam_2;
  real y_p;
}
transformed data {
  real transformed_data_real;
 
  transformed_data_real <- log_mix(d_real_theta,
                                         d_real_lam_1,
                                         d_real_lam_1);
}
parameters {
  real p_real_theta;
  real p_real_lam_1;
  real p_real_lam_2;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- log_mix(p_real_theta,
                                    p_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real <- log_mix(p_real_theta,
                                    d_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real <- log_mix(p_real_theta,
                                    d_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real <- log_mix(d_real_theta,
                                    p_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real <- log_mix(d_real_theta,
                                    p_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real <- log_mix(d_real_theta,
                                    d_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real <- log_mix(p_real_theta,
                                    p_real_lam_1,
                                    p_real_lam_1);
}
model {  
  y_p ~ normal(0,1);
}
