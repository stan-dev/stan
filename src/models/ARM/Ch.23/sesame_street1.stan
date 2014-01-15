data {
  int<lower=0> J;
  int<lower=0> N;
  int<lower=1,upper=J> siteset[N];
  vector[2] yt[N];
  vector[N] z;
}
parameters {
  vector[2] ag[J];
  real b;
  real d;
  real<lower=-1,upper=1> rho_ag;
  real<lower=-1,upper=1> rho_yt;
  vector[2] mu_ag;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_g;
  real<lower=0,upper=100> sigma_t;
  real<lower=0,upper=100> sigma_y;
}
model {
  vector[J] a;
  vector[J] g;
  matrix[2,2] Sigma_ag;
  matrix[2,2] Sigma_yt;
  vector[2] yt_hat[N];

  //data level
  Sigma_yt[1,1] <- pow(sigma_y,2);
  Sigma_yt[2,2] <- pow(sigma_t,2);
  Sigma_yt[1,2] <- rho_yt*sigma_y*sigma_t;  
  Sigma_yt[2,1] <- Sigma_yt[1,2];
   
  // group level
  Sigma_ag[1,1] <- pow(sigma_a,2);
  Sigma_ag[2,2] <- pow(sigma_g,2);
  Sigma_ag[1,2] <- rho_ag*sigma_a*sigma_g;
  Sigma_ag[2,1] <- Sigma_ag[1,2];  

  for (j in 1:J) {
    a[j] <- ag[j,1];
    g[j] <- ag[j,2];
  }

  for (i in 1:N) {
    yt_hat[i,2] <- g[siteset[i]] + d * z[i];
    yt_hat[i,1] <- a[siteset[i]] + b * yt[i,2];
  }

  //data level
  sigma_y ~ uniform (0, 100);               
  sigma_t ~ uniform (0, 100);                  
  rho_yt ~ uniform(-1, 1);
  d ~ normal (0, 31.6);
  b ~ normal (0, 31.6);

  //group level
  sigma_a ~ uniform (0, 100);
  sigma_g ~ uniform (0, 100);
  rho_ag ~ uniform(-1, 1);
  mu_ag ~ normal (0, 31.6);

  for (j in 1:J)
    ag[j] ~ multi_normal(mu_ag,Sigma_ag);

  //data model
  for (i in 1:N)
    yt[i] ~ multi_normal(yt_hat[i],Sigma_yt);

}
