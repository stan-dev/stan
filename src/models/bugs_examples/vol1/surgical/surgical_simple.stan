data {
    int N;
    int r[N];
    int n[N];
}
parameters {
    real<lower=0,upper=1> p[N];
}
model {
  p ~ beta(1.0, 1.0); 
  r ~ binomial(n, p);
}
