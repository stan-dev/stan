data {
    int<lower=0> N;
    int<lower=0> n[N];
    int<lower=0> r[N];
    real x[N];
}
transformed data {
    real mean_x;
    vector[N] centered_x;

    mean_x <- mean(x);
    for (i in 1:N)
        centered_x[i] <- x[i] - mean_x;
}
parameters {
    real alpha_star;
    real beta;
}
transformed parameters {
    vector[N] m;
    m <- alpha_star + beta * centered_x;

}


model {
  alpha_star ~ normal(0.0, 1.0E4);	
  beta ~ normal(0.0, 1.0E4);
  r ~ binomial_logit(n, m);
}

generated quantities {
  real alpha; 
  real p[N];
  real llike[N];
  real rhat[N];
  for (i in 1:N)  {
    p[i] <- inv_logit(m[i]);
    // log likelihood for sample i & saturated log-likelihood:
    llike[i]  <- r[i]*log(p[i]) + (n[i]-r[i])*log(1-p[i]);  
    // llike.sat[i] <- r[i]*log(r[i]/n[i]) + (n[i]-r[i])*log(1-r[i]/n[i]);
     rhat[i] <- p[i]*n[i];  // fitted values
  }
  alpha <- alpha_star - beta*mean_x;              
} 


