data {

}

transformed data {
  vector[5] theta_0;  // initial guess
  vector[2] phi;
  vector[5] x[2];
  int n_samples[5];
  int sums[5];
  vector[5] ye;
  real tol;
  int max_steps;

  real marginal_dbl;
  marginal_dbl = laplace_marginal_bernoulli(theta_0, phi, x, n_samples, sums, tol, max_steps);
  marginal_dbl = laplace_marginal_poisson(theta_0, phi, x, n_samples, sums, tol, max_steps);
  marginal_dbl = laplace_marginal_poisson(theta_0, phi, x, n_samples, sums, ye, tol, max_steps);
}

parameters {
  vector[5] theta_0_v;
  vector[2] phi_v;
  real dummy_parameter;
}

transformed parameters {
  real marginal;

  // logistic Bernoulli likelihood
  marginal = laplace_marginal_bernoulli(theta_0, phi, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_bernoulli(theta_0_v, phi, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_bernoulli(theta_0, phi_v, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_bernoulli(theta_0_v, phi_v, x, n_samples, sums, tol, max_steps);

  // log Poisson likelihood
  marginal = laplace_marginal_poisson(theta_0, phi, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0_v, phi, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0, phi_v, x, n_samples, sums, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0_v, phi_v, x, n_samples, sums, tol, max_steps);

  // log Poisson likelihood with exposure term
  marginal = laplace_marginal_poisson(theta_0, phi, x, n_samples, sums, ye, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0_v, phi, x, n_samples, sums, ye, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0, phi_v, x, n_samples, sums, ye, tol, max_steps);
  marginal = laplace_marginal_poisson(theta_0_v, phi_v, x, n_samples, sums, ye, tol, max_steps);
}

model {
  dummy_parameter ~ normal(0, 1);
}
