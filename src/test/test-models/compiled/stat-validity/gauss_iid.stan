transformed data {
  int<lower=0> N;
  N <- 5;
}

parameters {
  real x[N];
}

transformed parameters {
  real x2[N];
  for (n in 1:N)
    x2[n] <- x[n] * x[n];
}

model {
  x ~ normal(0, 1);
}