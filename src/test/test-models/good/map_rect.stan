functions {
  vector foo(vector shared_params, vector job_params,
             real[] data_r, int[] data_i) {
    return [1, 2, 3]';
  }
  real map_rectfake(real x) {
    return 2 * x;
  }
}
data {
  vector[3] shared_params_d;
  vector[3] job_params_d[3];
  real data_r[3, 3];
  int data_i[3, 3];
}
parameters {
  vector[3] shared_params_p;
  vector[3] job_params_p[3];
}
transformed parameters {
  real abc1_p = 3;
  real abc2_p = map_rectfake(abc1_p);
  real abc3_p = map_rectfake(12);
  vector[3] y_hat_tp1
      = map_rect(foo, shared_params_p, job_params_d, data_r, data_i);
  vector[3] y_hat_tp2
      = map_rect(foo, shared_params_d, job_params_p, data_r, data_i);
  vector[3] y_hat_tp3
      = map_rect(foo, shared_params_p, job_params_d, data_r, data_i);
}
model {
  real abc_m = map_rectfake(abc1_p);
}
generated quantities {
  real abc1_gq = map_rectfake(12);
  real abc2_gq = map_rectfake(abc1_p);
  vector[3] y_hat_gq
      = map_rect(foo, shared_params_d, job_params_d, data_r, data_i);
}
