functions {
  vector foo(vector shared_params, vector job_params,
             real[] data_r, int[] data_i) {
    return [1, 2, 3]';
  }
}
data {
  vector[3] shared_params_d;
  vector[3] job_params_d[3];
  real data_r[3, 3];
  int data_i[3, 3];
}
parameters {
  real a;
  vector[3] shared_params_p;
  vector[3] job_params_p[3];
  real data_r_p[3, 3];
}
generated quantities {
  vector[3] y_hat_gq
      = map_rect(foo, shared_params_d, job_params_d, data_r,
                 { { 1,  a > 2 ? 1 : 2 } });
}
