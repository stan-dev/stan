transformed data {
    real mu;
    real sigma;
    
    mu <- -20.0;
    sigma <- 10.0;
}
parameters {
    real y;
}
model {
    y ~ normal(mu,sigma);
}
