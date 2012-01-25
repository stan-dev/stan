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
    int(0,) I;
    int(0,) n[I];
    int(0,) N[I];
    double x1[I];
    double x2[I];
} 

transformed data {
    int K; 
    K <- 8; // {1, 2, 4, 8, 16, 32, 64, 128, 256}
} 

parameters {
    double alpha0;
    double alpha1;
    double alpha2;
    double alpha12;
    double(0,) tau;
    double b[I, K];
} 

transformed parameters {
    double sigma; 
    sigma <- 1 / sqrt(tau); 
} 

model {  
   alpha0 ~ normal(0.0, 1.0E3);
   alpha1 ~ normal(0.0, 1.0E3);
   alpha2 ~ normal(0.0, 1.0E3);
   alpha12 ~ normal(0.0, 1.0E3);
   tau ~ gamma(1.0E-3, 1.0E-3);
   for (i in 1:I) {
       for (k in 1:K) { 
           b[i, k] ~ normal(0.0, sigma);
           n[i] ~ binomial(N[i], inv_logit(alpha0 
                                           + alpha1 * x1[i] 
                                           + alpha2 * x2[i]
                                           + alpha12 * x1[i] * x2[i] 
                                           + b[i, k]));
       }
   }
} 
