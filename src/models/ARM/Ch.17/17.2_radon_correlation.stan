data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  int county[N];
  vector[N] u;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real g_a_0;
  real g_a_1;
  real g_b_0;
  real g_b_1;
  real<lower=-1,upper=1> rho;
  vector[2] B_temp;
}
model {
  vector[N] y_hat;
  vector[J] a;
  vector[J] b;
  matrix[J,2] B_hat;
  matrix[2,2] Sigma_b;
  matrix[J,2] B;

  g_a_0 ~ normal(0, 100);
  g_a_1 ~ normal(0, 100);
  g_b_0 ~ normal(0, 100);
  g_b_1 ~ normal(0, 100);
  rho ~ uniform(-1, 1);

  Sigma_b[1,1] <- pow(sigma_a, 2);
  Sigma_b[2,2] <- pow(sigma_b, 2);
  Sigma_b[1,2] <- rho * sigma_a * sigma_b;
  Sigma_b[2,1] <- Sigma_b[1,2];

  for (j in 1:J) {
    B_hat[j,1] <- g_a_0 + g_a_1 * u[j];
    B_hat[j,2] <- g_b_0 + g_b_1 * u[j];
    B_temp ~ multi_normal(transpose(row(B_hat,j)),Sigma_b);
    B[j,1] <- B_temp[1];
    B[j,2] <- B_temp[2];
    a[j] <- B[j,1];
    b[j] <- B[j,2];
  }

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]] * x[i];

  y ~ normal(y_hat, sigma);
}
