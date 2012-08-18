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

    for (i in 1:N)
      p[i] <- Phi(alpha_star + beta*centered_x[i]);
}
model {
    alpha_star ~ normal(0.0, 1.0);	
    beta ~ normal(0.0, 1.0E4);
    for (i in 1:N)
        r[i] ~ binomial(n[i], p[i]);
}
generated quantities {
//  real D;
  real alpha; 
  real llike[N];
//  real llike_sat[N];
  real rhat[N];

  alpha <- alpha_star - beta*mean_x;              

  for (i in 1:N) {
      llike[i]  <- r[i] * log(p[i]) 
                   + (n[i] - r[i]) * log(1-p[i]);  
//      llike_sat[i] <- r[i] * log(r[i] / n[i]) 
//                      + (n[i] - r[i]) * log(1-r[i] / n[i]);
      rhat[i] <- p[i] * n[i]; 
   }
//   D <- 2 * (sum(llike_sat) - sum(llike));
}
