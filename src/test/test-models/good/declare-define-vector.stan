data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  vector[7] foo;
  vector[7] bar = foo;
  row_vector[7] baz;
  {
    vector[7] loc_td_b1 = bar;
  }
}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  vector[7] tpar_b = bar;
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  vector[7] gq_b1 = bar;
  row_vector[7] gq_c1 = baz;
  {
    vector[7] loc_gq_b1 = bar;
    row_vector[7] loc_gq_c1 = baz;
  }
}
