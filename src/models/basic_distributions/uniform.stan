parameters {
    real<lower=0,upper=1> y;
}
model {
    y ~ uniform(0,1);
}
