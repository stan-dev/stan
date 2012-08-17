data {
    int<lower=0> N;
    int<lower=0> n[N];
    int<lower=0> r[N];
    real x[N];
}
transformed data {
    real mean_x;
    real centered_x[N];

    mean_x <- mean(x);
    for (i in 1:N)
        centered_x[i] <- x[i] - mean_x;
}
parameters {
    real alpha_star;
    real beta;
}
transformed parameters {
    real p[N];
    real llike[N];
    real rhat[N];


    for (i in 1:N) {
        p[i] <- 1.0 - inv_cloglog(alpha_star + beta*centered_x[i]);
        // log likelihood for sample i & saturated log-likelihood:
        llike[i]  <- r[i]*log(p[i]) + (n[i]-r[i])*log(1-p[i]);  
        // llike.sat[i] <- r[i]*log(r[i]/n[i]) + (n[i]-r[i])*log(1-r[i]/n[i]);
        rhat[i] <- p[i]*n[i];  // fitted values
    }
    //D <- 2 * (sum(llike.sat[]) - sum(llike[]));
}


model {
   alpha_star ~ normal(0.0, 1.0E4);	
   beta ~ normal(0.0, 1.0E4);
   for (i in 1:N)
      r[i] ~ binomial(n[i], p[i]);
}


generated quantities {
  real alpha; 
  alpha <- alpha_star - beta*mean_x;              
} 

