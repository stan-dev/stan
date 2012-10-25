// doesn't work because truncation not vectorized

parameters {
    real y[10];
}
model { 
    y ~ normal(0.0,1.0) T[0.0,1.1];
}
