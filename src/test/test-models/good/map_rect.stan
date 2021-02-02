functions {
  vector foo(vector shared_params, vector job_params, array[] real data_r,
             array[] int data_i) {
    return [1, 2, 3]';
  }
  real map_rectfake(real x) {
    return 2 * x;
  }
}
data {
  vector[3] shared_params_d;
  array[3] vector[3] job_params_d;
  array[3, 3] real data_r;
  array[3, 3] int data_i;
}
parameters {
  vector[3] shared_params_p;
  array[3] vector[3] job_params_p;
}
transformed parameters {
  real abc1_p = 3;
  real abc2_p = map_rectfake(abc1_p);
  real abc3_p = map_rectfake(12);
  vector[3] y_hat_tp1 = map_rect(foo, shared_params_p, job_params_d, data_r,
                                 data_i);
  vector[3] y_hat_tp2 = map_rect(foo, shared_params_d, job_params_p, data_r,
                                 data_i);
  vector[3] y_hat_tp3 = map_rect(foo, shared_params_p, job_params_d, data_r,
                                 data_i);
}
model {
  real abc_m = map_rectfake(abc1_p);
}
generated quantities {
  real abc1_gq = map_rectfake(12);
  real abc2_gq = map_rectfake(abc1_p);
  vector[3] y_hat_gq = map_rect(foo, shared_params_d, job_params_d, data_r,
                                data_i);
}

