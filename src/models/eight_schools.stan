data {
    int J(0,);
    double(0,) sigma_y[J];
    double y[J];
}
parameters {
    double mu_theta;
    double xi;
    double eta[J];
}
derived {
    double(0,) sigma_theta;
}
model {
    sigma_eta ~ inverse_gamma(0.5, 0.5);
    for (j in 1:J) 
        eta[j] ~ normal(0, sigma_eta);

    mu_theta ~ normal(0,10);
    xi ~ normal(0, prior_scale);
    for (j in 1:J)
        y[j] ~ normal(mu_theta + xi * eta[j], sigma_y[j]);

    sigma_theta <- abs(xi) * sigma_eta;
}