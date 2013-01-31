data {
  int<lower=0> T;
  int<lower=0> I;
  int<lower=0> K;
  matrix[T,I] X;
  real<lower=0> sigma[I];
}
parameters {
  matrix<lower=0>[T,K] G;
  simplex[I] F[K];
}
model {
  for (t in 1:T) {
    for (i in 1:I) {
      real mu;
      mu <- 0;
      for (k in 1:K) {
        mu <- mu + G[t,k] * F[k,i];
      }
      X[t,i] ~ normal(mu,sigma[i]);
    }
  }
}
