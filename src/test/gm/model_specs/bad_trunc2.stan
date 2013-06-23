// doesn't work because no CDF for uniform

parameters {
    real y;
}
model { 
    y ~ uniform(0.0,1.0) T[0.0,1.1];
}
