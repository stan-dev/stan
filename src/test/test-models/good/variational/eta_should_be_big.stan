transformed data {
    vector[2]   y[2];
    vector[2]   mu0;
    matrix[2,2] sigma;

    y[1,1]  <- 40000.0;
    y[1,2]  <- 30000.0;

    y[2,1]  <- 30000.0;
    y[2,2]  <- 20000.0;

    mu0[1]  <- 0.0;
    mu0[2]  <- 0.0;

    sigma[1,1] <- 10000.0;
    sigma[1,2] <- 00000.0;
    sigma[2,1] <- 00000.0;
    sigma[2,2] <- 10000.0;
}
parameters {
    vector[2] mu;
}
model {
    mu ~ multi_normal(mu0,sigma);

    for (n in 1:2)
        y[n] ~ multi_normal(mu,sigma);
}
