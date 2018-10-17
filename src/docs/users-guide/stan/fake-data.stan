data {
  int<lower=1> N;
  real a;
  real b;
  real<lower=0> sigma;
  vector[N] x;
}
generated quantities {
  vector[N] y;
  for (n in 1:N)
    y[n] = normal_rng(a + b * x[n], sigma);
}
