data {
    double(0,) sigma;
    int N;
    int x[N];
    vector(4) aa;
    matrix(3,4) bb[5,6];
}
derived data {
    int y[N];
    vector(3) a;
    vector(4) b[5];
    vector(4) c[5,6];
    matrix(3,4) d;
    matrix(3,4) e[5];
    matrix(3,4) f[5,6];

    for (n in 1:N)
        y[n] <- log(x[n]);
}
parameters {
    double mu;
}
derived parameters {
    double two_mu;
    two_mu <- mu * 2;
}
model {
   for (n in 1:N) 
       y[n] ~ normal(mu,sigma);
}
