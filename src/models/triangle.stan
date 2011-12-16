parameters {
    double(-1,1) y;
}
model {
    lp__ <- lp__ + log(1 - abs(y));
}