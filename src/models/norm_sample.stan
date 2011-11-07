data {
    double mu;
    double(0,) sigma;
}
derived data {
    double mu_pp;

    mu_pp <- mu + 1;
    mu_pp <- 2 * mu_pp;
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
