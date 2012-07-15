##  http://www.openbugs.info/Examples/Salm.html

## the version without centering x's 
data {
    int(0,)  doses;
    int(0,)  plates;
    int(0,)  y[doses,plates];
    real(0,) x[doses];
}
parameters {
    real alpha; 
    real beta;
    real gamma;
    real(0,) tau;
    real lambda[doses,plates];
}
transformed parameters {
    real(0,) sigma;
    sigma <- 1.0 / sqrt(tau);
}
model {
   alpha ~ normal(0.0, 100);
   beta ~ normal(0.0, 100);
   gamma ~ normal(0.0, 1.0E5); 
   tau ~ gamma(0.001, 0.001);
   for (dose in 1:doses) {
      for (plate in 1:plates) {
         lambda[dose, plate] ~ normal(0.0, sigma);
         y[dose, plate] ~ poisson(exp(alpha +
                                      beta * log(x[dose] + 10) +
                                      gamma * x[dose] +
                                      lambda[dose, plate]) );
     }
   }
}
