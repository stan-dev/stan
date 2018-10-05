transformed data {
  real alpha1 = 0.2;
  real alpha2 = 0.2;
  real beta1 = 1;
  real beta2 = 1;
  real t1 = 0.281152420352801;
  real t2 = 0.164745791682175;
}
parameters {
  real<lower=0> lambda1;
  real<lower=0> lambda2;
}
transformed parameters {
  real nu = lambda1 / 2.0 + lambda2;
}
model {
  lambda1 ~ gamma(alpha1, beta1);
  lambda2 ~ gamma(alpha2, beta2);
  t1 ~ exponential(lambda1);
  t2 ~ exponential(nu);
}
