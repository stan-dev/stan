##  http://www.openbugs.info/Examples/Salm.html
##  this matches the jags implementation
data {
    int<lower=0> Ndoses;
    int<lower=0> Nplates;
    int<lower=0> y[Ndoses,Nplates];
    real x[Ndoses];
}
transformed data {
    real logx[Ndoses];
    real mean_x;
    real mean_logx;
    real centered_x[Ndoses];
    real centered_logx[Ndoses];

    mean_x <- mean(x);
    for (dose in 1:Ndoses)
        centered_x[dose] <- x[dose] - mean_x;

    for (dose in 1:Ndoses)
        logx[dose] <- log(x[dose] + 10);
    mean_logx <- mean(logx);
    for (dose in 1:Ndoses)
        centered_logx[dose] <- logx[dose] - mean_logx;
}
parameters {
    real alpha_star;
    real beta;
    real gamma;
    real<lower=0> tau;
    vector[Nplates] lambda[Ndoses];
}
transformed parameters {
    real<lower=0> sigma;
    real alpha;

    alpha <- alpha_star - beta * mean_logx - gamma * mean_x;
    sigma <- 1.0 / sqrt(tau);
}

model {
   alpha_star ~ normal(0.0,1.0E3);
   beta ~ normal(0.0,1000);
   gamma ~ normal(0.0,1000); 
   tau ~ gamma(0.001,0.001);
   for (dose in 1:Ndoses) {
      lambda[dose] ~ normal(0.0, sigma); 
      y[dose] ~ poisson_log(alpha_star 
                            + beta * centered_logx[dose]
                            + gamma * centered_x[dose]
                            + lambda[dose]);
   }
}
