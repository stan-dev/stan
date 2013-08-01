data {
  int<lower=0> N;
  int<lower=0> J; 
  vector[N] y;
  vector[N] y_hat;

  vector[J] g;
} 
parameters {
  real<lower=0> sigma_g;
}
transformed parameters {
  vector[N] e_y;
  vector[J] e_g;
  real<lower=0> s_y;
  real<lower=0> s_g;
  real<lower=0> s_d;
  real<lower=0> s_g2;
  vector[J] g_hat;

  //finite population sd
  e_y <- y - y_hat;
  s_y <- sd(e);
  s_g <- sd(g);
  s_d <- sd(d);

  //finite population sd for when group level predictors are present
  g_hat <- a_0 + a_1 * u_1 + a_2 * u;
  e_g <- g - g_hat;
  s_g2 <- sd(e_g);
} 
model {
  sigma_g ~ uniform(0, 100);

  g ~ normal(g_hat, sigma_g)
}
