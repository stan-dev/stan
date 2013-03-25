data {
  int<lower=0> T;
  int<lower=0> I;
  int<lower=0> K;
  matrix[T,I] X;
  real<lower=0> sigma[I];
}
transformed data {
  real<lower=0> g_bar;
  real<lower=0> g_sigma;
  vector<lower=0>[I] alpha;
  vector[T] temp;
  for (t in 1:T) 
    temp[t] <- log(sum(X[t]));
  g_bar <- mean(temp);
  g_sigma <- sd(temp);
  for (i in 1:I)
    alpha[i] <- 10.0;
}
parameters {
  matrix<lower=0>[T,K] G;
  simplex[I] F[K];         // implicit unif prior on simplexes
}
model {
  for (t in 1:T)
    G[t] ~ lognormal(g_bar,g_sigma);
  for (k in 1:K)
    F[k] ~ dirichlet(alpha);

  for (t in 1:T) {
    for (i in 1:I) {
      real mu;
      mu <- 0;
      for (k in 1:K)
        mu <- mu + G[t,k] * F[k,i];
      X[t,i] ~ normal(mu,sigma[i]) T[0,];
    }
  }
}
