data {
    int N;
    int r[N];
    int n[N];
}
parameters {
    real<lower=0,upper=1> p[N];
}
model {
  for (i in 1:N) {
    p[i] ~ beta(1.0, 1.0);
    r[i] ~ binomial(n[i], p[i]);
  }
}
