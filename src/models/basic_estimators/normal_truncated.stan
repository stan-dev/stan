data {
  real U;
  int<lower=1> N;
  real<upper=U> y[N];
}
parameters {
  real mu;
  real<lower=0,upper=2> sigma;
}
model {
  for (n in 1:N)
    y[n] ~ normal(mu,sigma) T[,U];
}
