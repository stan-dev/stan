parameters {
    double x;
}
derived parameters {
    double(,0) lp;
model {
    x ~ normal(0,1);
    lp <- lp__;
}
generated quantities {
    double(,0) lp;
}