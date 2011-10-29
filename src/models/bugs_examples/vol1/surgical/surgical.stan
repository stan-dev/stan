data {
    int N;
    int r[N];
    int n[N];
}
parameters {
     double b[N];
//     double(0,1) p[N];
     double mu;
     double(0,) sigma;
     double pop_mean;
}
model {
  for (i in 1:N) {
    b[i] ~ normal(mu, sigma);
    r[i] ~ binomial(n[i], inv_logit(b[i]));
  }
  pop_mean <- exp(mu) / (1 + exp(mu));
  mu ~ normal(0.0, 1.0E6);
  sigma ~ inv_gamma(0.001,0.001);
}

