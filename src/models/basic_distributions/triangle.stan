parameters {
    real(-1,1) y;
}
model {
    lp__ <- log1m(fabs(y));
}
