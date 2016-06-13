functions {
  real foo_lpdf(real y, real mu) {
    return -(y - mu)^2;
  }

  real foo_lcdf(real y, real mu) {
    return -1.7;
  }

  real foo_lccdf(real y, real mu) {
    return -0.02;
  }
}
parameters {
  real<lower=1, upper=5> y;
  real mu;
}
model {
  y ~ foo(mu); 
  y ~ foo(mu) T[1, ];
  y ~ foo(mu) T[, 5];
  y ~ foo(mu) T[1, 5];
}
