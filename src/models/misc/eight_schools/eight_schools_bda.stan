// EIGHT SCHOOLS MODEL

data {
    int[0,] J;             // number of schools
    real y[J];             // estimated treatment effect (school j)
    real[0,] sigma_y[J];   // std dev of effect estimate (school j)
}
parameters {
    real theta[J];     
    real mu_theta;   
    real[0,1000] sigma_theta; 
}
model {
    // loop
    //    for (j in 1:J)
    //        theta[j] ~ normal(mu_theta, sigma_theta);
    //    for (j in 1:J)
    //        y[j] ~ normal(theta[j], sigma_y[j]);
  
    // vectorized loops  
    mu_theta ~ normal(0,1000);
    theta ~ normal(mu_theta, sigma_theta); 
    y ~ normal(theta,sigma_y);
}
