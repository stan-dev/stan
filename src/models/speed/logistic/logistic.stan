data {
  int(0,) N;
  int(0,) M;
  int(0,1) y[N];
  real x[N,M];
}
parameters {
  real alpha;
  real beta[M];
}
model {
  alpha ~ normal(0.0, 100);
  beta ~ normal(0.0, 100);
  
  for (i in 1:N) {
    real p;
    p <- alpha;
    for (j in 1:M) {
      p <- p + x[i,j] * beta[j];
    }
    y[i] ~ binomial(1, inv_logit(p));
  }
}