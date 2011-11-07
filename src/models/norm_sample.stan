data {
    double mu;
    double(0,) sigma;
}
derived data {
    double mu_pp;
    mu_pp <- mu + 1;
}
parameters {
    double y;
}
derived {
    double z;
}
model {
    y ~ normal(mu,sigma);
}
