data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  matrix[2,3] m23;
  matrix[2,3] yam23 = m23;
}
parameters {
  real<lower=0,upper=1> theta;
} 
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
