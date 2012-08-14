// EIGHT SCHOOLS MODEL
// ported from:
// Gelman et al., Bayesian Data Analysis, 2nd Edition, p. 598
// EIGHT SCHOOLS MODEL

data {
    int[0,] J;               // number of schools
    real y[J];             // estimated treatment effect (school j)
    real[0,] sigma_y[J];   // standard error of effect estimate (school j)
    real sigma_xi;         // prior scale (coefficient)
}
parameters {
    real mu;               // intercept coefficient (for y)
    real xi;               // slope coefficient (for y)
    real eta[J];           // predictor (school j)
    real[0,] sigma_eta;    // deviation of eta
}
model {
    sigma_eta ~ cauchy(0,1);
    for (j in 1:J)
        eta[j] ~ normal(0, sigma_eta);
    mu ~ normal(0,10);
    xi ~ normal(0, sigma_xi);
    for (j in 1:J)
        y[j] ~ normal(mu + xi * eta[j], sigma_y[j]);
}
