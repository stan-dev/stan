// Test both lgp solvers: first one is for the case where the
// latent Gaussian variable has a diagonal covariance matrix,
// second case generalises this to a dense covariance matrix.
// This are experiment is rather specific.

data {

}

transformed data {
 vector[5] theta_0;  // initial guess
 real phi;  // global parameter
 vector[2] phi_vec;  // global parameter in vector form (for general case)
 int n_samples[5];  // number of terms for a local parameter
 int samples[5];  // sum of observations for a local parameter
 int max_num_steps;
 int is_line_search;
 int print_iteration;
 int space_matters;
 real tol;

 vector[5] theta_dbl;
 theta_dbl = lgp_newton_solver(theta_0, phi, n_samples, samples);
 theta_dbl = lgp_dense_newton_solver(theta_0, phi_vec, n_samples, samples);
}

parameters {
  vector[5] theta_0_v;
  real phi_v;
  vector[2] phi_vec_v;
  real dummy_parameter;
}

transformed parameters {
  vector[5] theta;

  // lgp_conditional
  theta = lgp_newton_solver(theta_0, phi, n_samples, samples);
  theta = lgp_newton_solver(theta_0_v, phi, n_samples, samples);
  theta = lgp_newton_solver(theta_0, phi_v, n_samples, samples);
  theta = lgp_newton_solver(theta_0_v, phi_v, n_samples, samples);

  theta = lgp_newton_solver(theta_0, phi, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0_v, phi, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0, phi_v, n_samples, samples, tol, max_num_steps, is_line_search);
  theta = lgp_newton_solver(theta_0_v, phi_v, n_samples, samples, tol, max_num_steps, is_line_search);
  
  // lgp_dense
  theta = lgp_dense_newton_solver(theta_0, phi_vec, n_samples, samples);
  theta = lgp_dense_newton_solver(theta_0_v, phi_vec, n_samples, samples);
  theta = lgp_dense_newton_solver(theta_0, phi_vec_v, n_samples, samples);
  theta = lgp_dense_newton_solver(theta_0_v, phi_vec_v, n_samples, samples);

  theta = lgp_dense_newton_solver(theta_0, phi_vec, n_samples, samples, tol, max_num_steps, is_line_search, print_iteration, space_matters);
  theta = lgp_dense_newton_solver(theta_0_v, phi_vec, n_samples, samples, tol, max_num_steps, is_line_search, print_iteration, space_matters);
  theta = lgp_dense_newton_solver(theta_0, phi_vec_v, n_samples, samples, tol, max_num_steps, is_line_search, print_iteration, space_matters);
  theta = lgp_dense_newton_solver(theta_0_v, phi_vec_v, n_samples, samples, tol, max_num_steps, is_line_search, print_iteration, space_matters);
}

model {
  dummy_parameter ~ normal(0, 1);
}
