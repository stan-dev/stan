derived data {
    int N;
 
    N <- 10;
}
parameters {
    double y;
    double z[N];
}
model {
    y ~ normal(0,1);
    for (n in 1:N)
        z[n] ~ normal(-10,5);
}