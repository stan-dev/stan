data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  vector[7] foo;
  vector[7] bar = foo;
}
parameters {
  real<lower=0,upper=1> theta;
} 
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
