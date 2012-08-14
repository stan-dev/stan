data {
    int[0,] J;             // number of schools
    real y[J];             // estimated treatment effect (school j)
    real[0,] sigma[J];     // std dev of effect estimate (school j)
}
parameters {
    real mu;
    real theta[J];
    real[0,] tau;
}
model {
    theta ~ normal(mu, tau); 
    y ~ normal(theta,sigma);
}
