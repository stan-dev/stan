data {
  int N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real<lower=0> a;
  real<lower=0> b;
  real<lower=0> sigma;
}
model {
  y ~ normal(a*exp(-b*x), sigma);
  a ~ normal(0, 10);
  b ~ normal(0, 10);
  sigma ~ normal(0, 10);
}
