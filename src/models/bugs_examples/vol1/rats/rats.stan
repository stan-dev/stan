# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 3: Rats
data {
  int(0,) N;
  int(0,) T;
  double x[T];
  double xbar;
  double y[N,T];
}
parameters {
  double alpha[N];
  double beta[N];

  double mu_alpha;
  double mu_beta;

  double(0,) sigma_y;
  double(0,) sigma_alpha;
  double(0,) sigma_beta;
}
derived parameters {
  double(0,) sigmasq_y;
  double(0,) sigmasq_alpha;
  double(0,) sigmasq_beta;

  sigmasq_y <- sigma_y * sigma_y;
  sigmasq_alpha <- sigma_alpha * sigma_alpha;
  sigmasq_beta <- sigma_beta * sigma_beta;
}
model {
  mu_alpha ~ normal(0, 100);
  mu_beta ~ normal(0, 100);
  sigmasq_y ~ inv_gamma(0.001, 0.001);
  sigmasq_alpha ~ inv_gamma(0.001, 0.001);
  sigmasq_beta ~ inv_gamma(0.001, 0.001);
  alpha ~ normal(mu_alpha, sigma_alpha); // vectorized
  beta ~ normal(mu_beta, sigma_beta);  // vectorized
  for (n in 1:N)
    for (t in 1:T) 
      y[n,t] ~ normal(alpha[n] + beta[n] * (x[t] - xbar), sigma_y);

}
