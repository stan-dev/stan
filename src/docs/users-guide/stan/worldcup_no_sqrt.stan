/* This program has a mistake in it, as will be explained later */

data {
  int N_teams;
  int N_games;
  vector[N_teams] prior_score;
  int team_1[N_games];
  int team_2[N_games];
  vector[N_games] score_1;
  vector[N_games] score_2;
  real df;
}
transformed data {
  vector[N_games] dif;
  dif = score_1 - score_2;
}
parameters {
  vector[N_teams] alpha;
  real b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[N_teams] a;
  a = b*prior_score + sigma_a*alpha;
}
model {
  alpha ~ normal(0, 1);
  dif ~ student_t(df, a[team_1] - a[team_2], sigma_y);
}
generated quantities {
  vector[N_games] y_rep;
  for (n in 1:N_games) {
    y_rep[n] = student_t_rng(df, a[team_1[n]] - a[team_2[n]], sigma_y);
  }
}
