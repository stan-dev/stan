data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  int county[N];
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real mu_a;
  real mu_b;
  real<lower=-1,upper=1> rho;
  vector[2] B_temp;
}
model {
  vector[N] y_hat;
  vector[J] a;
  vector[J] b;
  matrix[2,J] B_hat;
  matrix[2,2] Sigma_b;
  vector[2] B_hat_temp;
  matrix[2,J] B;

  mu_a ~ normal(0, 100);
  mu_b ~ normal(0, 100);
  rho ~ uniform(-1, 1);

  Sigma_b[1,1] <- pow(sigma_a, 2);
  Sigma_b[2,2] <- pow(sigma_b, 2);
  Sigma_b[1,2] <- rho * sigma_a * sigma_b;
  Sigma_b[2,1] <- Sigma_b[1,2];

  for (j in 1:J) {
    B_hat[1,j] <- mu_a;
    B_hat[2,j] <- mu_b;
    B_hat_temp[1] <- mu_a;
    B_hat_temp[2] <- mu_b;
    B_temp ~ multi_normal(B_hat_temp, Sigma_b);
    B[1,j] <- B_temp[1];
    B[2,j] <- B_temp[2];
  }

  for (j in 1:J) {
    a[j] <- B[1,j];
    b[j] <- B[2,j];
  }

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]] * x[i];

  y ~ normal(y_hat, sigma);
}
