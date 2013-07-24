## 
##  http://www.openbugs.info/Examples/Seeds.html
data {
    int<lower=0> I;
    int<lower=0> n[I];
    int<lower=0> N[I];
    vector[I] x1; 
    vector[I] x2; 
}

transformed data {
    vector[I] x1x2;
    x1x2 <- x1 .* x2;
} 
parameters {
    real alpha0;
    real alpha1;
    real alpha12;
    real alpha2;
    real<lower=0> tau;
    vector[I] b;
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
   n ~ binomial_logit(N, alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b);
}
