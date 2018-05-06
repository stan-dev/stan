data {

}

transformed data {
 vector[5] theta_0;  // initial guess
 real phi;  // global parameter
 int n_samples[5];  // number of terms for a local parameter
 int samples[5];  // sum of observations for a local parameter
 int max_num_steps;
int is_line_search;
 real tol;

 vector[5] theta_dbl;
 theta_dbl = lgp_newton_solver(theta_0, phi, n_samples, samples);
}

parameters {
  vector[5] theta_0_v;
  real phi_v;
  real dummy_parameter;
}

transformed parameters {
  vector[5] theta;

  theta = lgp_newton_solver(theta_0, phi, n_samples, samples);
  theta = lgp_newton_solver(theta_0_v, phi, n_samples, samples);
  theta = lgp_newton_solver(theta_0, phi_v, n_samples, samples);
  theta = lgp_newton_solver(theta_0_v, phi_v, n_samples, samples);

  theta = lgp_newton_solver(theta_0, phi, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0_v, phi, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0, phi_v, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0_v, phi_v, n_samples, samples, tol, max_num_steps, is_line_search);
}

model {
  dummy_parameter ~ normal(0, 1);
}
