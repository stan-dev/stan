data {
  int<lower=2> K;               // num topics
  int<lower=2> V;               // num words
  int<lower=1> M;               // num docs
  int<lower=1> N;               // total word instances
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  simplex[K] theta;   // topic prevalence
  simplex[V] phi[K];  // word dist for topic k
}
model {
  real gamma[M,K];

  theta ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);

   for (m in 1:M) 
     for (k in 1:K) 
       gamma[m,k] <- categorical_log(k,theta);
   for (n in 1:N)
     for (k in 1:K)
       gamma[doc[n],k] <- gamma[doc[n],k] 
                          + categorical_log(w[n],phi[k]);
   for (m in 1:M)
     increment_log_prob(log_sum_exp(gamma[m]));

   // to normalize s.t. gamma[m,k] = log Pr[Z2[m] = k|data]
   // gamma[m] <- gamma[m] - log_sum_exp(gamma[m]);
}
