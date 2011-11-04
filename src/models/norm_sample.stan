data {
    double mu;
    double(0,) sigma;
}
parameters {
    double y;
}
derived {
    double w[2];
    double z[5,7];
}
model {
    y ~ normal(mu,sigma);
}
