functions {
  real foo(real a1) {
    real lf1 = a1;
    print(" in function foo");
    print("    lf1: ",lf1);
    print("    a1: ",a1);
    lf1 *= a1;
    print("    lf1 *= a1: ",lf1);
    return lf1;
  }
}
data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  int x1 = 1;
  real y1 = 2;
  print("in transformed data");
  print("x1: ", x1, " y1: ", y1);
  x1 *= 1;  // scalar int
  y1 *= 1;  // scalar double
  print("x1 *= 1: ", x1, " y1 *= 1: ", y1);
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  real w = 7;
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
  print("in model block");
  print("w: ", w);
  w *= theta; // scalar var
  print("w *= theta: ", w);
  w *= foo(theta); // scalar var
  print("w *= foo(theta): ", w);
}
generated quantities {
  real z = 1;
  print("in generated quantities");
  print("theta ", theta);
  print("z: ", z);
  print("y1: ", y1);
  z *= theta + y1;
  print("z *= theta + y1: ", z);
  z *= foo(theta) + foo(y1);
  print("z *= foo(theta) + foo(y1): ", z);
}
