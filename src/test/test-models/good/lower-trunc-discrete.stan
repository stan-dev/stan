functions {
  real foo_lpmf(int y, real lambda) {
    return 1.0;
  }
  real foo_lcdf(int y, real lambda) {
    return 1.0;
  }
  real foo_lccdf(int y, real lambda) {
    return 1.0;
  }
}

data {
  int N;
  int y[N];
  int L;
  int U;
}
parameters {
  real<lower=0> lambda;
}
model {
  for (n in 1:N) {
    y[n] ~ poisson(lambda) T[L, ];
    y[n] ~ poisson(lambda) T[L, U];
    y[n] ~ poisson(lambda) T[ , U];

    y[n] ~ foo(lambda) T[L, ];
    y[n] ~ foo(lambda) T[L, U];
    y[n] ~ foo(lambda) T[ , U];
  }
}
