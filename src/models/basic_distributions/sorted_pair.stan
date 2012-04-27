parameters {
    real(-1,1) x1;
    real(-1,1) x2;
}
model {
}
generated quantities {
    real a;
    real b;
    a <- fmax(x1,x2);
    b <- fmin(x1,x2);
}