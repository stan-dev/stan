data {
    real y;
}

model {
    y ~ normal(0,1) T[-1,1];
    y ~ normal(0,1) T[0, ];
    y ~ normal(0,1) T[ ,0];
    y ~ normal(0,1);

    for (n in 1:5) ;

    lp__ <- lp__ + 1.0; // lp__ is type real
}