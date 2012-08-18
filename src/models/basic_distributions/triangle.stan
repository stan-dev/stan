parameters {
    real<lower=-1,upper=1> y;
}
model {
    lp__ <- lp__ + log1m(fabs(y));
}
