data {
  int<lower=1> K;  // num categories
  int<lower=1> V;  // num words
  int<lower=0> N;  // num instances
  int<lower=1,upper=V> w[N]; // words
  int<lower=1,upper=K> z[N]; // categories
  vector<lower=0>[K] alpha;  // transit prior
  vector<lower=0>[V] beta;   // emit prior
}
parameters {
  simplex[K] theta[K];  // transit probs
  simplex[V] phi[K];    // emit probs
}
model {
  for (k in 1:K) 
    theta[k] ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  for (n in 1:N)
    w[n] ~ categorical(phi[z[n]]);
  for (n in 2:N)
    z[n] ~ categorical(theta[z[n-1]]);
}
