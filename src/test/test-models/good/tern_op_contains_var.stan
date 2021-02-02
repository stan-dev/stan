data {
  int N;
  int y_max;
  array[N] int<lower=0> y;
  array[N] int<lower=0, upper=1> group;
}
parameters {
  vector<lower=0, upper=1>[2] param;
}
model {
  to_vector(param) ~ uniform(0, 1);
  for (i in 1 : N) {
    y[i] ~ binomial(y_max, group[i] == 0 ? param[1] : param[2]);
  }
}

