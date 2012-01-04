parameters {
    double(-1,1) y;
}
model {
    lp__ <- lp__ + log(fmax(0.0,1.0 - fabs(y))); 
}
