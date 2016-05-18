data {
    real y;
}

model {
    y ~ normal(0,1) T[-1,1];
    y ~ normal(0,1) T[0, ];
    y ~ normal(0,1) T[ ,0];
    y ~ normal(0,1);
}
