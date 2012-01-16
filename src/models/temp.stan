parameters {
    double y;
    double z[2];
}
derived parameters {
   double y_adj;

   y_adj <- fabs(y);
}
model {
    y ~ normal(0,1); 
    for (n in 1:2)
        z[n] ~ normal(-10,5);
}
generated quantities {
   double z_adj[2];

   for (n in 1:2) 
      z_adj[n] <- z[n] * z[n];
}