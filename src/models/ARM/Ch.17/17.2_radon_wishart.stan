data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  vector[N] u;
  int county[N];
  matrix[2,2] W;
}
parameters {
  real<lower=0> sigma;
  real mu_a_raw;
  real mu_b_raw;
  real<lower=0> xi_a;
  real<lower=0> xi_b;
  vector[2] B_raw_temp;
  corr_matrix[2] Tau_b_raw;
}
model {
  vector[N] y_hat;
  vector[J] a;
  vector[J] b;
  matrix[J,2] B_raw_hat;
  matrix[J,2] B_raw;
  matrix[2,2] Sigma_b_raw;
  real g_a_0;
  real g_a_1;
  real g_b_0;
  real g_b_1;
  real sigma_a;
  real sigma_b;
  real rho;

  mu_a_raw ~ normal(0, 100);
  mu_b_raw ~ normal(0, 100);
  xi_a ~ uniform(0, 100);
  xi_b ~ uniform(0, 100);

  g_a_0 <- xi_a * mu_a_raw;
  g_a_1 <- xi_a * mu_a_raw;
  g_b_0 <- xi_b * mu_b_raw;
  g_b_1 <- xi_b * mu_b_raw;

  Tau_b_raw ~ wishart(3, W);
  Sigma_b_raw <- inverse(Tau_b_raw);

  sigma_a <- xi_a * sqrt(Sigma_b_raw[1,1]);
  sigma_b <- xi_b * sqrt(Sigma_b_raw[2,2]);
  rho <- Sigma_b_raw[1,2] / sqrt(Sigma_b_raw[1,1] * Sigma_b_raw[2,2]);

  for (j in 1:J) {
    B_raw_hat[j,1] <- g_a_0 + g_a_1 * u[j];
    B_raw_hat[j,2] <- g_b_0 + g_b_1 * u[j];
    B_raw_temp ~ multi_normal(transpose(row(B_raw_hat,j)), Sigma_b_raw);
    B_raw[j,1] <- B_raw_temp[1];
    B_raw[j,2] <- B_raw_temp[2];
    a[j] <- xi_a * B_raw[j,1];
    b[j] <- xi_b * B_raw[j,2];
  }

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]] * x[i];

  y ~ normal(y_hat, sigma);
}
