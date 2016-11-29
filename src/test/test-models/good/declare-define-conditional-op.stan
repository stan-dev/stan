data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
parameters {
  real<lower=0,upper=1> theta;
  real z;
}
transformed parameters {
  real y1 = 1 ? y[1] : z;
  {
    real loc_y1 = 1 ? y[1] : z;
  }
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  real y2 = 1 ? y[2] : z;
}

