data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
parameters {
  real<lower=0,upper=1> theta;
} 
model {
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
