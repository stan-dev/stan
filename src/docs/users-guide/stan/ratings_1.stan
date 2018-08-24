data {
  int N;
  vector[N] y;
  int<lower=1, upper=2> movie[N];
}
parameters {
  vector<lower=0, upper=5>[2] theta;
  real<lower=0, upper=2.5> sigma_y;
}
model {
  theta ~ normal(3, 1);
  y ~ normal(theta[movie], sigma_y);
  /* equivalently:
    for (j in 1:2){
      theta[j] ~ normal(3, 1);
    }
    for (n in 1:N){
      y[n] ~ normal(theta[movie[n]], sigma_y);
    }
  */
}

