# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 3: Rats
data {
  int<lower=0> N;
  int<lower=0> T;
  vector[T] x;
  matrix[N,T] y;
  real xbar;
}
parameters {
  vector[N] alpha_z;
  vector[N] beta_z;

  real mu_alpha;
  real mu_beta;

  real<lower=0> sigmasq_y;
  real<lower=0> sigmasq_alpha;
  real<lower=0> sigmasq_beta;
}
transformed parameters {
  real<lower=0> sigma_y;       // sigma in original bugs model
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[N] alpha;
  vector[N] beta;

  sigma_y <- sqrt(sigmasq_y);
  sigma_alpha <- sqrt(sigmasq_alpha);
  sigma_beta <- sqrt(sigmasq_beta);

  alpha <- (mu_alpha * 100.0) + alpha_z * sigma_alpha;
  beta <- (mu_beta * 100.0) + beta_z * sigma_beta;
}
model {
  mu_alpha ~ normal(0,1);
  mu_beta ~ normal(0,1);
  sigmasq_y ~ inv_gamma(0.001, 0.001);
  sigmasq_alpha ~ inv_gamma(0.001, 0.001);
  sigmasq_beta ~ inv_gamma(0.001, 0.001);
  alpha_z ~ normal(0, 1);
  beta_z ~ normal(0,1); 
  for (t in 1:T)
    col(y,t) ~ normal(alpha + beta * (x[t] - xbar), sigma_y);

}
generated quantities {
  real alpha0;
  alpha0 <- mu_alpha - xbar * mu_beta;
}
