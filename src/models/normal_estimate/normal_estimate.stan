data {
     int(0,) N;
     double y[N];
}
parameters {
     double mu;
     double(0,) sigma;
}
model {
    for (n in 1:N) {
        y[n] ~ normal(mu,sigma);
    }
}

