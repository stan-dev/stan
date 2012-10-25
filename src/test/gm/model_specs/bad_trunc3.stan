// doesn't work because truncation not vectorized

parameters {
    real y;
    real mu[10];
}
model { 
    y ~ normal(mu,1.0) T[0.0,1.1];
}
