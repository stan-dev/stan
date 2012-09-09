data {
  int<lower=1> J;             // number of students
  int<lower=1> K;             // number of questions
  int<lower=1> N;             // number of observations
  int<lower=1,upper=J> jj[N];  // student for observation n
  int<lower=1,upper=K> kk[N];  // question for observation n
  int<lower=0,upper=1> y[N];  // correctness of observation n
}
parameters {
  vector<J> alpha;
  vector<K> beta;
}
model {
  alpha ~ normal(0,10);
  beta ~ normal(0,10);
  for (n in 1:N)
    y[n] ~ bernoulli_logit(alpha[jj[n]] - beta[kk[n]]);
}
