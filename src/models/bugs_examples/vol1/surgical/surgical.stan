data {
    int N;
    int r[N];
    int n[N];
}
parameters {
     double b[N];
     double mu;
     double(0,) sigma;
}
derived parameters {
    double pop_mean;
    pop_mean <- exp(mu) / (1.0 + exp(mu)); // fixme -- inv-logit
}
model {
  for (i in 1:N) {
    b[i] ~ normal(mu, sigma);
    r[i] ~ binomial(n[i], inv_logit(b[i]));
  }
  mu ~ normal(0.0, 1.0E6);
  sigma ~ inv_gamma(0.001,0.001);
}

