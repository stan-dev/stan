##  http://www.openbugs.info/Examples/Salm.html
##  this matches the jags implementation
data {
    int<lower=0> doses;
    int<lower=0> plates;
    int<lower=0> y[doses,plates];
    real x[doses];
}
transformed data {
    real logx[doses];
    real mean_x;
    real mean_logx;
    real centered_x[doses];
    real centered_logx[doses];

    mean_x <- mean(x);
    for (dose in 1:doses)
        centered_x[dose] <- x[dose] - mean_x;

    for (dose in 1:doses)
        logx[dose] <- log(x[dose] + 10);
    mean_logx <- mean(logx);
    for (dose in 1:doses)
        centered_logx[dose] <- logx[dose] - mean_logx;
}
parameters {
    real alpha_star;
    real beta;
    real gamma;
    real<lower=0> tau;
    real lambda[doses,plates];
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
   for (dose in 1:doses) {
      for (plate in 1:plates) {
         lambda[dose,plate] ~ normal(0.0, sigma); 
         y[dose,plate] ~ poisson(exp(alpha_star 
                              + beta * centered_logx[dose]
                              + gamma * centered_x[dose]
                              + lambda[dose,plate]) );
     }
   }
}
