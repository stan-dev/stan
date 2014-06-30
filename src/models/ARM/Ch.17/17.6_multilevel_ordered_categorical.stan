data {
  int<lower=0> N;
  int<lower=0> n_cut;
  int<lower=0> n_player;

  int<lower=0, upper=n_player> player[N];
  int<lower=0, upper=n_cut> y[N];
  vector[N] x;
}
parameters {
  vector[n_cut] mu_c;
  real mu_log_s;

  vector<lower=0>[n_cut] sigma_c;
  real<lower=0> sigma_log_s;

  matrix[n_player,2] C;
  vector[n_player] s;
}
model {
  real mu_adj;
  matrix[N,n_cut] Q;
  matrix[N,n_cut] P;

  mu_c ~ normal(0, 1000);
  mu_log_s ~ normal(0, 100);

  for (i in 1:n_player) {
    C[i,1] ~ normal(mu_c[1], sigma_c[1]) T[0,C[i,2]];
    C[i,2] ~ normal(mu_c[2], sigma_c[1]) T[C[i,1],100];
    s[i] ~ lognormal(mu_log_s, sigma_log_s) T[1,100];
  }

  for (i in 1:N) {
    for (i_cut in 1:n_cut)
      Q[i,i_cut] <- inv_logit((x[i] - C[player[i],i_cut])/s[player[i]]);

    P[i,1] <- 1 - Q[i,1];
    P[i,n_cut+1] <- Q[i,n_cut];
    for (i_cut in 2:n_cut)
      P[i,i_cut] <- Q[i,i_cut-1] - Q[i,i_cut];
  }

  for (i in 1:N)
    y[i] ~ categorical(transpose(row(P,i)));

}
