data {
  int<lower=0> J;
  int<lower=0> N;
  vector[N] pretest;
  int<lower=1,upper=J> siteset[N];
  matrix[N,2] yt;
  vector[N] z;
}
parameters {
  matrix[J,2] ag;
  real b;
  real d;
  vector[2] mu_ag;
  real phi_y;
  real phi_t;
  real<lower=-1,upper=1> rho_ag;
  real<lower=-1,upper=1> rho_yt;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_g;
  real<lower=0,upper=100> sigma_py;
  real<lower=0,upper=100> sigma_pt;
  real<lower=0,upper=100> sigma_t;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[J] a;
  vector[J] g;
  cov_matrix[2] Sigma_ag;
  cov_matrix[2] Sigma_yt;
  matrix[N,2] yt_hat;

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
    yt_hat[i,1] <- a[siteset[i]] + b * d * z[i] + phi_y * pretest[i];
    yt_hat[i,2] <- g[siteset[i]] + d * z[i] + phi_t * pretest[i];
  }
}
model {
  //data level
  sigma_y ~ uniform (0, 100);               
  sigma_t ~ uniform (0, 100);                  
  rho_yt ~ uniform(-1, 1);
  d ~ normal (0, 100);
  b ~ normal (0, 100);
  phi_y ~ normal(0, sigma_py);
  phi_y ~ normal(0, sigma_pt);

  //group level
  sigma_a ~ uniform (0, 100);
  sigma_g ~ uniform (0, 100);
  rho_ag ~ uniform(-1,1);
  mu_ag[1] ~ normal (0, 100);
  mu_ag[2] ~ normal (0, 100);

  //data model
  for (i in 1:N)
    transpose(yt[i]) ~ multi_normal(transpose(yt_hat[i]),Sigma_yt);

  for (j in 1:J)
    transpose(ag[j]) ~ multi_normal(mu_ag,Sigma_ag);
}
