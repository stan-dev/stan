## 
##  http://www.openbugs.info/Examples/Seeds.html
data {
    int<lower=0> I;
    int<lower=0> n[I];
    int<lower=0> N[I];
    real x1[I];
    real x2[I];
}
parameters {
    real alpha0;
    real alpha1;
    real alpha12;
    real alpha2;
    real<lower=0> tau;
    real b[I];
}
transformed parameters {
    real<lower=0> sigma;
    sigma  <- 1.0 / sqrt(tau);
}
model {
   alpha0 ~ normal(0.0,1.0E3);
   alpha1 ~ normal(0.0,1.0E3);
   alpha2 ~ normal(0.0,1.0E3);
   alpha12 ~ normal(0.0,1.0E3);
   tau ~ gamma(1.0E-3,1.0E-3);

   b ~ normal(0.0, sigma);

   for (i in 1:I) {
      n[i] ~ binomial(N[i], inv_logit(alpha0 
                                      + alpha1 * x1[i] 
                                      + alpha2 * x2[i]
                                      + alpha12 * x1[i] * x2[i] 
                                      + b[i]) );
   }
}
