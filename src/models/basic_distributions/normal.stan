transformed data {
    double mu;
    double sigma;
    
    mu <- -20.0;
    sigma <- 10.0;
}
parameters {
    double y;
}
model {
    y ~ normal(mu,sigma);
}
