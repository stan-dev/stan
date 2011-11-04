data {
    double mu;
    double(0,) sigma;
}
parameters {
    double y;
}
derived {
    double y;
}
model {
    y ~ normal(mu,sigma);
}
