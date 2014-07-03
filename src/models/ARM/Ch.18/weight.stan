data {
  int<lower=0> N;
  vector[N] weight;
  vector[N] height;
}
transformed data {           // centering height
  vector[N] c_height;
  c_height <- height - mean(height);
}
parameters {
  real a;
  real b;
  real<lower=0> sigma;
}
model {
  weight ~ normal(a + b * c_height, sigma);
}
