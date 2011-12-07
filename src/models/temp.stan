// EIGHT SCHOOLS MODEL

data {
    int(0,) J;               // number of schools
    double y[J];             // estimated treatment effect (school j)
    double(0,) sigma_y[J];   // standard error of effect estimate (school j)
    double sigma_xi;         // prior scale (coefficient)
}
parameters {
    double mu;               // intercept coefficient (for y)
    double xi;               // slope coefficient (for y)
    double eta[J];           // predictor (school j)
    double(0,) sigma_eta;    // deviation of eta
}
model {
    sigma_eta ~ cauchy(0,1);
    for (j in 1:J)
        eta[j] ~ normal(0, sigma_eta);
    mu ~ normal(0,10);
    xi ~ normal(0, sigma_xi);
    for (j in 1:J)
        y[j] ~ normal(eta, sigma_y[j]);
}
