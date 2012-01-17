data {
    int(0,) I;
    int(0,) J;
    int(0,) y[I,J];
    double x[I];
}
transformed data {
    double logx[I];
    double mean_x;
    double mean_logx;
    double centered_x[I];
    double centered_logx[I];

    mean_x <- mean(x);
    for (i in 1:I)
        centered_x[i] <- x[i] - mean_x;

    for (i in 1:I)
        logx[i] <- log(x[i] + 10);
    mean_logx <- mean(logx);
    for (i in 1:I)
        centered_logx[i] <- logx[i] - mean_logx;
}
parameters {
    double alpha_star;
    double beta;
    double gamma;
    double(0,) tau;
    double lambda[I,J];
}
transformed parameters {
    double sigma;
    double alpha;

    alpha <- alpha_star - beta * mean_logx - gamma * mean_x;
    sigma <- 1.0 / sqrt(tau);
}
model {
   alpha_star ~ normal(0.0,1.0E3);
   beta ~ normal(0.0,1.0E3);
   gamma ~ normal(0.0,1.0E3); 
   tau ~ gamma(1.0E-3,1.0E-3);
   for (i in 1:I) {
      for (j in 1:J) {
         lambda[i,j] ~ normal(0.0,tau);
         y[i,j] ~ poisson(exp(alpha_star 
                              + beta * centered_x[i]
                              + gamma * centered_logx[i]
                              + lambda[i,j]) );
     }
   }
}
