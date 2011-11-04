data {
    double mu;
    double(0,) sigma;
}
parameters {
    double y;
}
derived {
    double z;
}
model {
    y ~ normal(mu,sigma);
    for (n in) {
    }
}
