# Hearts: a mixture model for count data
# http://www.openbugs.info/Examples/Hearts.html
# 

## status: not work (there are discrete parameters) 
data {
  int(0,) N;
  int(0,) x[N];
  int(0,) y[N];
  int(0,) t[N];
} 

parameters {
  real alpha; 
  real delta; 
  int(0,) state[N]; // these are discrete parameters. the support is 0 or 1 actually. ??
} 

transformed parameters {
  real theta;
  theta <- inv_logit(delta); 
} 

model {
  real yap[2]; 
  yap[1] <- inv_logit(alpha); 
  yap[2] <- 0;

  alpha ~ normal(0, 100);
  delta ~ normal(0, 100); 
  for (i in 1:N) {
    state[i] ~ bernoulli(theta); // either 0 or 1
    y[i] ~ binomial(t[i], yap[state[i] + 1]);
  }
}
