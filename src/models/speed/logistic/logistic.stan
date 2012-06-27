data {
  int(0,) N;               // number of items
  int(0,) M;               // number of predictors
  int(0,1) y[M];           // outcomes
  row_vector(M) x[N];      // predictors
}
parameters {
  vector(M) beta;          // coefficients
}
model {
  for (m in 1:M)
    beta[m] ~ cauchy(0.0, 2.5);
  
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(x[n] * beta));
}