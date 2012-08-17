# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 3: Rats
data {
  int<lower=0> N;
  int<lower=0> T;
  real x[T];
  real y[N,T];
  real xbar;
}
transformed data {
  real x_minus_xbar[T];
  real y_linear[N*T];

  for (t in 1:T)
    x_minus_xbar[t] <- x[t] - xbar;

  for (n in 1:N)
    for (t in 1:T)
      y_linear[(n-1)*T + t] <- y[n, t];
}
parameters {
  real alpha[N];
  real beta[N];

  real mu_alpha;
  real mu_beta;

  real<lower=0> sigmasq_y;
  real<lower=0> sigmasq_alpha;
  real<lower=0> sigmasq_beta;
}
transformed parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;

  sigma_y <- sqrt(sigmasq_y);
  sigma_alpha <- sqrt(sigmasq_alpha);
  sigma_beta <- sqrt(sigmasq_beta);
}
model {
  real pred[N*T];

  for (n in 1:N)
    for (t in 1:T)
      pred[(n-1)*T + t] <- fma(beta[n], x_minus_xbar[t], alpha[n]);

  mu_alpha ~ normal(0, 100);
  mu_beta ~ normal(0, 100);
  sigmasq_y ~ inv_gamma(0.001, 0.001);
  sigmasq_alpha ~ inv_gamma(0.001, 0.001);
  sigmasq_beta ~ inv_gamma(0.001, 0.001);
  alpha ~ normal(mu_alpha, sigma_alpha); // vectorized
  beta ~ normal(mu_beta, sigma_beta);  // vectorized

  y_linear ~ normal(pred, sigma_y);  // vectorized
}
