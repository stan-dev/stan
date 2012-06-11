##  http://www.openbugs.info/Examples/Salm.html

## the version without centering x's 
data {
    int(0,) I;
    int(0,) J;
    int(0,) y[I,J];
    real x[I];
}
parameters {
    real alpha; 
    real beta;
    real gamma;
    real(0,) tau;
    real lambda[I,J];
}
transformed parameters {
    real(0,) sigma;
    sigma <- 1.0 / sqrt(tau);
}
model {
   alpha ~ normal(0.0, 1.0E3);
   beta ~ normal(0.0, 1.0E3);
   gamma ~ normal(0.0, 1.0E3); 
   tau ~ gamma(1.0E-3, 1.0E-3);
   for (i in 1:I) {
      for (j in 1:J) {
         lambda[i, j] ~ normal(0.0, sigma); 
         y[i, j] ~ poisson(exp(alpha 
                               + beta * log(x[i] + 10) 
                               + gamma * x[i] 
                               + lambda[i, j]) );
     }
   }
}
