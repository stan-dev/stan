data {
    int<lower=0> N;
    int<lower=0> n[N];
    int<lower=0> r[N];
    vector[N] x;
}
transformed data {
    vector[N] centered_x;
    real mean_x;
    mean_x <- mean(x);
    centered_x <- x - mean_x;
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
    r ~ binomial(n, p);
}
generated quantities {
  real alpha; 
  real llike[N];
  real rhat[N];

  alpha <- alpha_star - beta*mean_x;              

  for (i in 1:N) {
    llike[i]  <- r[i] * log(p[i]) + (n[i] - r[i]) * log(1-p[i]);
    rhat[i] <- p[i] * n[i]; 
  }
}
