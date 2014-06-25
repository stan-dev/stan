data {
  int<lower=0> G;
  int<lower=0> N;
  int<lower=0> r[G,N];
  int<lower=0> n[G,N];
}
parameters {
  matrix<lower=0,upper=1>[G,N] p;
  vector<lower=0.1>[G] a;
  vector<lower=0.1>[G] b;
}
model {
  a ~ gamma(1,0.001);
  b ~ gamma(1,0.001);
  for (g in 1:G) {
    for (i in 1:N) { 
      p[g,i] ~ beta(a[g],b[g]);
      r[g,i] ~ binomial(n[g,i],p[g,i]);
    }
  }      
}
generated quantities {
  vector<lower=0,upper=1>[G] mu;
  vector<lower=0>[G] theta;
  for (g in 1:G)
    mu[g] <- a[g] / (a[g] + b[g]);
  for (g in 1:G)
    theta[g] <- 1 / (a[g] + b[g]);
}