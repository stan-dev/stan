// poisson_log doesn't have cdf and ccdf functions
data {
  int n;
  int L;
  int U;
}
parameters {
  real alpha;
}
model {
  n ~ poisson_log(alpha) T[L, U];
}
