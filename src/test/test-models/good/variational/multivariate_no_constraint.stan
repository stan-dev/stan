transformed data {
    vector[2]   y[2];
    vector[2]   mu0;
    matrix[2,2] sigma;

    y[1,1]  <- 4.0;
    y[1,2]  <- 3.0;

    y[2,1]  <- 3.0;
    y[2,2]  <- 2.0;

    mu0[1]  <- 3.5;
    mu0[2]  <- 2.5;

    sigma[1,1] <- 1.0;
    sigma[1,2] <- 0.0;
    sigma[2,1] <- 0.0;
    sigma[2,2] <- 1.0;
}
parameters {
    vector[2] mu;
}
model {
    mu ~ multi_normal(mu0,sigma);

    for (n in 1:2)
        y[n] ~ multi_normal(mu,sigma);
}
