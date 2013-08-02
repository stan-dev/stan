# Using Data Cloning to Calculate MLEs for the Seeds Model in vol1 
# http://www.openbugs.info/Examples/DataCloning.html

# The basic idea is that we raise the likelihood in the 
# posterior to the power of K so that the posterior
# estimates would concentrate on the MLE estimates. 
# Reference: 
#   Ecology Letters
#   Subhash R. Lele Brian Dennis Frithjof Lutscher
#   DOI: 10.1111/j.1461-0248.2007.01047.x
#   http://onlinelibrary.wiley.com/doi/10.1111/j.1461-0248.2007.01047.x/abstract


data {
    int<lower=0> I;
    int<lower=0> n[I];
    int<lower=0> N[I];
    vector[I] x1;
    vector[I] x2;
} 

transformed data {
    int K; 
    vector[I] x1x2;
    K <- 8; // {1, 2, 4, 8, 16, 32, 64, 128, 256}
    x1x2 <- x1 .* x2;
} 

parameters {
    real alpha0;
    real alpha1;
    real alpha2;
    real alpha12;
    real<lower=0> tau;
    vector[K] b[I];
} 

transformed parameters {
    real sigma; 
    sigma <- 1 / sqrt(tau); 
} 

model {  
   alpha0 ~ normal(0.0, 1.0E3);
   alpha1 ~ normal(0.0, 1.0E3);
   alpha2 ~ normal(0.0, 1.0E3);
   alpha12 ~ normal(0.0, 1.0E3);
   tau ~ gamma(1.0E-3, 1.0E-3);
   for (i in 1:I) {
      b[i] ~ normal(0.0, sigma);
      n[i] ~ binomial_logit(N[i], alpha0 + alpha1 * x1[i] + alpha2 * x2[i] + alpha12 * x1x2[i] + b[i]);
   }
} 
