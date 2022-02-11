data {
  int<lower=0> N;
  int<lower=0> M;
  array[N] int<lower=0, upper=1> y;
  array[N] row_vector[M] x;
}
parameters {
  vector[M] beta;
}
model {
  for (m in 1 : M) 
    beta[m] ~ cauchy(0.0, 2.5);
  for (n in 1 : N) 
    y[n] ~ bernoulli(inv_logit(x[n] * beta));
}

