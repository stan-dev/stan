transformed data {
    vector[2]   y;
    matrix[2,2] sigma;

    y[1]  <- 0.0;
    y[2]  <- 0.0;

    sigma[1,1] <- 1.0;
    sigma[1,2] <- 0.0;
    sigma[2,1] <- 0.0;
    sigma[2,2] <- 1.0;
}
parameters {
    vector[2] mu;
}
model {
    y ~ multi_normal(mu,sigma);
}
