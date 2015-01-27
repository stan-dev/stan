data {
  int<lower=0> N;               // number of items
  int<lower=0> M;               // number of predictors
  int<lower=0,upper=1> y[N];           // outcomes
  row_vector[M] x[N];      // predictors
}
parameters {
  vector[M] beta;          // coefficients
}
model {
  for (m in 1:M)
    beta[m] ~ cauchy(0.0, 2.5);
  
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(x[n] * beta));
}
