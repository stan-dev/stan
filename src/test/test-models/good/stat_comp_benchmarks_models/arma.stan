data {
  int<lower=1> T;
  array[T] real y;
}
parameters {
  real mu;
  real phi;
  real theta;
  real<lower=0> sigma;
}
model {
  vector[T] nu;
  vector[T] err;
  mu ~ normal(0, 10);
  phi ~ normal(0, 2);
  theta ~ normal(0, 2);
  sigma ~ cauchy(0, 2.5);
  nu[1] = mu + phi * mu;
  err[1] = y[1] - nu[1];
  for (t in 2 : T) {
    nu[t] = mu + phi * y[t - 1] + theta * err[t - 1];
    err[t] = y[t] - nu[t];
  }
  err ~ normal(0, sigma);
}

