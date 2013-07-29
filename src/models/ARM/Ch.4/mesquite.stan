data {
  int<lower=0> N;
  vector[N] weight;
  vector[N] diam1;
  vector[N] diam2;
  vector[N] canopy_height;
  vector[N] total_height;
  vector[N] density;
  vector[N] group;
}
parameters {
  vector[7] beta;
  real<lower=0> sigma;
}
model {
  weight ~ normal(beta[1] + beta[2] * diam1 + beta[3] * diam2
                  + beta[4] * canopy_height + beta[5] * total_height
                  + beta[6] * density + beta[7] * group,
                  sigma);
}
