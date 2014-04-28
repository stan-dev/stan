functions {
  void linear_regression_lp(vector x, vector y, 
                            real alpha, real beta, real sigma) {
    y ~ normal(x * alpha + beta, sigma);
    sigma ~ cauchy(0,2.5);
    alpha ~ normal(0,10);
    beta ~ normal(0,10);
  }
}
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  linear_regression_lp(x,y,alpha,beta,sigma);
}
