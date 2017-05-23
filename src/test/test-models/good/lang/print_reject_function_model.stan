functions {
  void foo(real x) {
    print("quitting time");
    reject("user-specified rejection");
  }
}
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
  real v;
  print("In model block.");
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
  foo(v)
}
generated quantities {
  int z;
  print("In generated quantities");
  z = N;
}
