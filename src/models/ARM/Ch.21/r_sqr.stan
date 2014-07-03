data {
  int<lower=0> N;
  int<lower=0> n_county;
  int<lower=1,upper=n_county> county[N];
  vector[N] u;
  vector[N] y;
} 
parameters {
  vector[n_county] a;
  vector[n_county] b;
  matrix[j,2] B;
  matrix[j,2] B_hat;
  vector[2] B_hat_temp;
  vector[2] B_temp;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]]*x[i];

  e_y <- y - y_hat; //data-level errors
} 
model {
  sigma_y ~ uniform(0, 100);

  y ~ normal(y_hat, sigma_y);

  for (j in 1:n_county) {
    B_hat_temp[1] <- g_a_0 + g_a_1 * u[j];
    B_hat[j,1] <- B_hat_temp[1];
    B_hat_temp[2] <- g_b_0 + g_b_1 * u[j];
    B_hat[j,2] <- B_hat_temp[2];
    B_temp ~ multi_normal(b_hat_temp,sigma_B);
    B_hat[j,1] <- B_temp[1]; 
    B_hat[j,2] <- B_temp[2];
    for (k in 1:2)
      E_B[j,k] = B[j,k]-B_hat[j,k]; //group-level errors
    } 
}
