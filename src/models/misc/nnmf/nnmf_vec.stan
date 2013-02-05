// Evan Ira Blumgart (2012) Hamiltonian Monte Carlo Estimation of
// PM2.5 Source Apportionment and Related Health Effects.
// M.A. Dissertation.  Department of Statistics. 
// The University of Auckland.

data {
  int<lower=0> T;
  int<lower=0> I;
  int<lower=0> K;
  matrix<lower=0.0>[T,I] X;
  real<lower=0> sigma[I];
}
transformed data {
  real<lower=0> g_bar;
  real<lower=0> g_sigma;
  vector[T] temp;
  for (t in 1:T) 
    temp[t] <- log(sum(X[t]));
  g_bar <- mean(temp);
  g_sigma <- sd(temp);
}
parameters {
  matrix<lower=0>[T,K] G;
  simplex[I] F[K];
}
model {
  for (t in 1:T)
    G[t] ~ lognormal(g_bar,g_sigma);

  for (t in 1:T) {
    vector[I] mu;
    for (i in 1:I) {
      mu[i] <- 0;
      for (k in 1:K) {
        mu[i] <- mu[i] + G[t,k] * F[k,i];
      }
    }
    X[t] ~ normal(mu,sigma);
  }
}
