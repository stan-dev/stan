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
derived parameters {
    double z;
    double w[1];

    z <- y * 2.0 + 1.0;
    w[0] <- pow(z,3.0);
}
model {
    y ~ normal(mu,sigma);
}
