// EIGHT SCHOOLS MODEL

data {
    int(0,) J;             // number of schools
    real y[J];             // estimated treatment effect (school j)
    real(0,) sigma_y[J];   // std dev of effect estimate (school j)
}
parameters {
    real theta[J];     
    real mu_theta;   
    real(0,) sigma_theta; 
}
model {
    mu_theta ~ normal(0,1000);

    sigma_theta ~ uniform(0,1000);
    for (j in 1:J)
        theta[j] ~ normal(mu_theta, sigma_theta);
    for (j in 1:J)
        y[j] ~ normal(theta[j], sigma_y[j]);
}
