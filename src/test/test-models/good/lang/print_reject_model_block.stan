data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  int x = N;
  print("In transformed data");
}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  print("In transformed parameters");
}
model {
  print("In model block.");
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
  print("quitting time");
  reject("QUIT");
}
generated quantities {
  int z;
  print("In generated quantities");
  z = N;
}
