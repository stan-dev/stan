data {
  int<lower=0> G;
  int<lower=0> N;
  int<lower=0> r[G,N];
  int<lower=0> n[G,N];
}
parameters {
  matrix<lower=0,upper=1>[G,N] p;
  vector<lower=0,upper=1>[G] mu;
  vector<lower=0.1>[G] a_plus_b;
}
transformed parameters {
  vector[G] a;
  vector[G] b;
  a <- mu .* a_plus_b;
  b <- (1 - mu) .* a_plus_b;
}
model {
  a_plus_b ~ pareto(0.1,1.5);
  for (g in 1:G) {
    for (i in 1:N) { 
      p[g,i] ~ beta(a[g],b[g]);
      r[g,i] ~ binomial(n[g,i],p[g,i]);
    }
  }      
}
generated quantities {
  vector<lower=0>[G] theta;
  for (g in 1:G)
    theta[g] <- 1 / (a[g] + b[g]);
}