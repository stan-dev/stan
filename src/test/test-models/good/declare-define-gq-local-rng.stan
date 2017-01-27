transformed data {
  int a = categorical_rng(rep_vector(0.1, 10));
  {
    int b = categorical_rng(rep_vector(0.1, 10));
  }
}
parameters {
  real y;
}
transformed parameters {
  {
    int k;
  }
}
model {
  y ~ normal(0,1);
}
generated quantities {
  {
    int y_tilde = categorical_rng(rep_vector(0.1, 10));
  }
}
