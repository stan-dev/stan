data {
  int<lower=0> n_dogs;
  int<lower=0> n_trials;
  int<lower=0, upper=1> y[n_dogs,n_trials];
}
parameters {
  real<lower=0,upper=100> sigma_b1;
  real<lower=0,upper=100> sigma_b2;  
  matrix[n_dogs,2] beta_neg;
  real<lower=-1,upper=1> rho_b;
  vector[2] mu_beta;
}
model {
  vector[n_dogs] beta1;
  vector[n_dogs] beta2;
  matrix[n_dogs,n_trials] n_avoid;
  matrix[n_dogs,n_trials] n_shock;
  matrix[n_dogs,n_trials] p;
  matrix[2,2] Sigma_b;

  sigma_b1 ~ uniform (0, 100);               
  sigma_b2 ~ uniform (0, 100);                  
  rho_b ~ uniform(-1, 1);
  mu_beta ~ normal(0, 100);

  Sigma_b[1,1] <- pow(sigma_b1,2);
  Sigma_b[2,2] <- pow(sigma_b2,2);
  Sigma_b[1,2] <- rho_b*sigma_b1*sigma_b2;
  Sigma_b[2,1] <- Sigma_b[1,2];  

  for (i in 1:n_dogs)
    transpose(beta_neg[i]) ~ multi_normal_prec(mu_beta,Sigma_b);

  for (j in 1:n_dogs) {
    n_avoid[j,1] <- 0;
    n_shock[j,1] <- 0;
    beta1[j] <- -exp(beta_neg[j,1]);
    beta2[j] <- -exp(beta_neg[j,2]);
    for (t in 2:n_trials) {
      n_avoid[j,t] <- n_avoid[j,t-1] + 1 - y[j,t-1];
      n_shock[j,t] <- n_shock[j,t-1] + y[j,t-1];
    }
    for (t in 1:n_trials) {
      p[j,t] <- inv_logit(beta1[j] * n_avoid[j,t] + beta2[j] * n_shock[j,t]);
      y[j,t] ~ bernoulli(p[j,t]);
    }
  }
}
