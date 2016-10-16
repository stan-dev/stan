parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  int n;
  real z;
  vector[3] theta;
  vector[3] alpha;
  vector[3] v;
  vector[3] mu;
  matrix[3,3] Sigma;
  matrix[3,3] L;
  int ns[3];

  n <- bernoulli_rng(0.5);
  n <- bernoulli_logit_rng(0.0);
  n <- binomial_rng(15,0.3);
  // n <- binomial_logit_rng(15,-1.2);
  n <- beta_binomial_rng(42, 0.3, 1.9);
  n <- hypergeometric_rng(5,4,9);
  n <- neg_binomial_rng(1.2,3.9);
  n <- neg_binomial_2_rng(1.2,3.9);
  n <- neg_binomial_2_log_rng(1.2,3.9);
  n <- ordered_logistic_rng(1.9,theta);
  n <- poisson_rng(2.7);
  n <- poisson_log_rng(2.7);

  n <- categorical_rng(theta);
  ns <- multinomial_rng(theta,20);

  z <- normal_rng(0,1);
  z <- exp_mod_normal_rng(0.2,1.9,1.5);
  z <- skew_normal_rng(1.1, 2.3, 9.8);
  z <- student_t_rng(3,1,2);
  z <- cauchy_rng(-1.0,2.9);
  z <- double_exponential_rng(1.0,3.8);
  z <- logistic_rng(1.0,3.8);
  z <- gumbel_rng(-1.0,2.31);
  z <- lognormal_rng(-1.0, 3.6);
  z <- chi_square_rng(4.1);
  z <- inv_chi_square_rng(4.1);
  z <- scaled_inv_chi_square_rng(1.0, 3.0);
  z <- exponential_rng(2.9);
  z <- gamma_rng(1.0, 3.0);
  z <- inv_gamma_rng(0.1,0.1);
  z <- von_mises_rng(1.0,2.0);
  z <- weibull_rng(1.0,2.0);
  z <- pareto_rng(0.1, 1.5);
  z <- pareto_type_2_rng(3.0, 2.0, 1.5);
  z <- beta_rng(110.0, 250.1);
  z <- uniform_rng(-1.0, 1.0);
  z <- rayleigh_rng(1.0);
  z <- frechet_rng(2.0, 3.2);

  theta <- dirichlet_rng(alpha);
  v <- multi_normal_rng(mu,Sigma);
  // v <- multi_normal_prec_rng(mu,Sigma);
  v <- multi_normal_cholesky_rng(mu,L);
  v <- multi_student_t_rng(3.0,mu,Sigma);
  Sigma <- wishart_rng(3.0,Sigma);
  Sigma <- inv_wishart_rng(3.0,Sigma);
  Sigma <- lkj_corr_rng(3,2.5);
  L <- lkj_corr_cholesky_rng(3,3.0);
  // DEPRECATE:  Sigma <- lkj_cov_rng(Sigma, mu, alpha, 2.0);




}
