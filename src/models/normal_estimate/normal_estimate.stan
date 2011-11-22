data {
     int(0,) N;
     double y[N];
}
parameters {
     double mu;
     double(0,) sigma;
}
model {
    mu ~ normal(0,1);
    sigma ~ cauchy(0,2);
    for (n in 1:N) {
        y[n] ~ normal(mu,sigma);
    }
}

