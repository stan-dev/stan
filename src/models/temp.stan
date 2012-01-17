data {
    int N;
    double(0,1) theta;
}
derived data {
    double(0,1) one_minus_theta;

    one_minus_theta <- 1.0 - theta;
}
parameters {
    double y[N];
}
derived parameters {
    double(0,) y_abs[N];

    for (n in 1:N)
        y_abs[n] <- fabs(y[n]);
}
model {
    for (n in 1:N) 
        y[n] ~ normal(0,theta); 
}
generated quantities {
   double(,0) z_neg[N];

   for (n in 1:N) 
      z_neg[n] <- -y_abs[n];
}