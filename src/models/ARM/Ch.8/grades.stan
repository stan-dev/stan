data {
  int<lower=0> N; 
  vector[N] final;
  vector[N] midterm;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  final ~ normal(beta[1] + beta[2] * midterm,sigma);
}
