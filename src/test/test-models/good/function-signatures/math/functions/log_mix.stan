data {
  int N;
  int M;
  real d_real_theta;
  real d_real_theta_arr[N];
  vector[N] d_vec_theta;
  row_vector[N] d_rowvec_theta;
  real d_real_lam_1;
  real d_real_lam_2;
  real d_real_lam_arr[N];
  vector[N] d_vec_lam;
  row_vector[N] d_rowvec_lam;
  vector[N] d_vec_lam_arr[M];
  row_vector[N] d_rowvec_lam_arr[M];
  real y_p;
}
transformed data {
  real transformed_data_real;

  transformed_data_real = log_mix(d_real_theta,
                                         d_real_lam_1,
                                         d_real_lam_1);

  transformed_data_real = log_mix(d_real_theta_arr,d_real_lam_arr);
  transformed_data_real = log_mix(d_real_theta_arr,d_vec_lam);
  transformed_data_real = log_mix(d_real_theta_arr,d_rowvec_lam);
  transformed_data_real = log_mix(d_real_theta_arr,d_vec_lam_arr);
  transformed_data_real = log_mix(d_real_theta_arr,d_rowvec_lam_arr);

  transformed_data_real = log_mix(d_vec_theta,d_real_lam_arr);
  transformed_data_real = log_mix(d_vec_theta,d_vec_lam);
  transformed_data_real = log_mix(d_vec_theta,d_rowvec_lam);
  transformed_data_real = log_mix(d_vec_theta,d_vec_lam_arr);
  transformed_data_real = log_mix(d_vec_theta,d_rowvec_lam_arr);

  transformed_data_real = log_mix(d_rowvec_theta,d_real_lam_arr);
  transformed_data_real = log_mix(d_rowvec_theta,d_vec_lam);
  transformed_data_real = log_mix(d_rowvec_theta,d_rowvec_lam);
  transformed_data_real = log_mix(d_rowvec_theta,d_vec_lam_arr);
  transformed_data_real = log_mix(d_rowvec_theta,d_rowvec_lam_arr);
}
parameters {
  real p_real_theta;
  real p_real_theta_arr[N];
  vector[N] p_vec_theta;
  row_vector[N] p_rowvec_theta;
  real p_real_lam_1;
  real p_real_lam_2;
  real p_real_lam_arr[N];
  vector[N] p_vec_lam;
  row_vector[N] p_rowvec_lam;
  vector[N] p_vec_lam_arr[M];
  row_vector[N] p_rowvec_lam_arr[M];
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = log_mix(p_real_theta,
                                    p_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real = log_mix(p_real_theta,
                                    d_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real = log_mix(p_real_theta,
                                    d_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real = log_mix(d_real_theta,
                                    p_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real = log_mix(d_real_theta,
                                    p_real_lam_1,
                                    d_real_lam_1);
  transformed_param_real = log_mix(d_real_theta,
                                    d_real_lam_1,
                                    p_real_lam_1);
  transformed_param_real = log_mix(p_real_theta,
                                    p_real_lam_1,
                                    p_real_lam_1);

  transformed_param_real = log_mix(d_real_theta_arr,p_real_lam_arr);
  transformed_param_real = log_mix(d_real_theta_arr,p_vec_lam);
  transformed_param_real = log_mix(d_real_theta_arr,p_rowvec_lam);
  transformed_param_real = log_mix(d_real_theta_arr,p_vec_lam_arr);
  transformed_param_real = log_mix(d_real_theta_arr,p_rowvec_lam_arr);

  transformed_param_real = log_mix(p_real_theta_arr,d_real_lam_arr);
  transformed_param_real = log_mix(p_real_theta_arr,d_vec_lam);
  transformed_param_real = log_mix(p_real_theta_arr,d_rowvec_lam);
  transformed_param_real = log_mix(p_real_theta_arr,d_vec_lam_arr);
  transformed_param_real = log_mix(p_real_theta_arr,d_rowvec_lam_arr);

  transformed_param_real = log_mix(p_real_theta_arr,p_real_lam_arr);
  transformed_param_real = log_mix(p_real_theta_arr,p_vec_lam);
  transformed_param_real = log_mix(p_real_theta_arr,p_rowvec_lam);
  transformed_param_real = log_mix(p_real_theta_arr,p_vec_lam_arr);
  transformed_param_real = log_mix(p_real_theta_arr,p_rowvec_lam_arr);

  transformed_param_real = log_mix(d_vec_theta,p_real_lam_arr);
  transformed_param_real = log_mix(d_vec_theta,p_vec_lam);
  transformed_param_real = log_mix(d_vec_theta,p_rowvec_lam);
  transformed_param_real = log_mix(d_vec_theta,p_vec_lam_arr);
  transformed_param_real = log_mix(d_vec_theta,p_rowvec_lam_arr);

  transformed_param_real = log_mix(p_vec_theta,d_real_lam_arr);
  transformed_param_real = log_mix(p_vec_theta,d_vec_lam);
  transformed_param_real = log_mix(p_vec_theta,d_rowvec_lam);
  transformed_param_real = log_mix(p_vec_theta,d_vec_lam_arr);
  transformed_param_real = log_mix(p_vec_theta,d_rowvec_lam_arr);

  transformed_param_real = log_mix(p_vec_theta,p_real_lam_arr);
  transformed_param_real = log_mix(p_vec_theta,p_vec_lam);
  transformed_param_real = log_mix(p_vec_theta,p_rowvec_lam);
  transformed_param_real = log_mix(p_vec_theta,p_vec_lam_arr);
  transformed_param_real = log_mix(p_vec_theta,p_rowvec_lam_arr);

  transformed_param_real = log_mix(d_rowvec_theta,p_real_lam_arr);
  transformed_param_real = log_mix(d_rowvec_theta,p_vec_lam);
  transformed_param_real = log_mix(d_rowvec_theta,p_rowvec_lam);
  transformed_param_real = log_mix(d_rowvec_theta,p_vec_lam_arr);
  transformed_param_real = log_mix(d_rowvec_theta,p_rowvec_lam_arr);

  transformed_param_real = log_mix(p_rowvec_theta,d_real_lam_arr);
  transformed_param_real = log_mix(p_rowvec_theta,d_vec_lam);
  transformed_param_real = log_mix(p_rowvec_theta,d_rowvec_lam);
  transformed_param_real = log_mix(p_rowvec_theta,d_vec_lam_arr);
  transformed_param_real = log_mix(p_rowvec_theta,d_rowvec_lam_arr);

  transformed_param_real = log_mix(p_rowvec_theta,p_real_lam_arr);
  transformed_param_real = log_mix(p_rowvec_theta,p_vec_lam);
  transformed_param_real = log_mix(p_rowvec_theta,p_rowvec_lam);
  transformed_param_real = log_mix(p_rowvec_theta,p_vec_lam_arr);
  transformed_param_real = log_mix(p_rowvec_theta,p_rowvec_lam_arr);
}
model {  
  y_p ~ normal(0,1);
}
