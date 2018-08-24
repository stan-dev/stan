data {
  int N;
  vector[N] x;
  vector[N] y;
}
parameters {
  vector<lower=0>[2] a;
  positive_ordered[2] b;
  real<lower=0> sigma;
}
model {
  vector[N] y_pred;
  y_pred = a[1]*exp(-b[1]*x) + a[2]*exp(-b[2]*x);
  y ~ lognormal(log(y_pred), sigma);
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  sigma ~ normal(0, 1);
}
