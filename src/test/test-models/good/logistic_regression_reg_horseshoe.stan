data {
  int<lower=0> n;
  int<lower=0> d;
  array[n] int<lower=0, upper=1> y;
  matrix[n, d] x;
  real<lower=0> scale_icept;
  real<lower=0> scale_global;
  real<lower=1> nu_global;
  real<lower=1> nu_local;
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
}
parameters {
  real beta0;
  vector[d] z;
  real<lower=0> tau;
  vector<lower=0>[d] lambda;
  real<lower=0> caux;
}
transformed parameters {
  vector[d] beta;
  {
    vector[d] lambda_tilde;
    real c = slab_scale * sqrt(caux);
    lambda_tilde = sqrt(c ^ 2 * square(lambda)
                        ./ (c ^ 2 + tau ^ 2 * square(lambda)));
    beta = z .* lambda_tilde * tau;
  }
}
model {
  z ~ std_normal();
  lambda ~ student_t(nu_local, 0, 1);
  tau ~ student_t(nu_global, 0, scale_global * 2);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  beta0 ~ normal(0, scale_icept);
  y ~ bernoulli_logit_glm(x, beta0, beta);
}

