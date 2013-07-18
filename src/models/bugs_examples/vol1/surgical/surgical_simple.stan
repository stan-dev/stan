data {
    int N;
    int r[N];
    int n[N];
}
parameters {
    real<lower=0,upper=1> p[N];
}
model {
  r ~ binomial(n, p);
}
