transformed data {
    real y[5];
    y[1] <- 2.0;
    y[2] <- 1.0;
    y[3] <- -0.5;
    y[4] <- 3.0;
    y[5] <- 0.25;
}
parameters {
    real mu;
}
model {
    for (n in 1:5)
        y[n] ~ normal(mu,1.0);
}