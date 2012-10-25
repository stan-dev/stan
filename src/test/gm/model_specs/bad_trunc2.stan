// doesn't work because no CDF for Cauchy

parameters {
    real y;
}
model { 
    y ~ cauchy(0.0,1.0) T[0.0,1.1];
}
