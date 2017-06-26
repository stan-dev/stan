data {
  int<lower=0> N;
}
transformed data {
  vector[N] y;
  for (n in 1:N)
    y[n] = normal_rng(0, 1);
  print(y);
}
parameters {
  real mu;
  real<lower = 0> sigma;
}
model {
  y ~ normal(mu, sigma);
}
generated quantities {
  real mean_y = mean(y);
  real sd_y = sd(y);
}
