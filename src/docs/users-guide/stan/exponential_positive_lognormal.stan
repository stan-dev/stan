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
  vector[N] y_pred;
  y_pred = a*exp(-b*x);
  y ~ lognormal(log(y_pred), sigma);
  a ~ normal(0, 10);
  b ~ normal(0, 10);
  sigma ~ normal(0, 10);
}
