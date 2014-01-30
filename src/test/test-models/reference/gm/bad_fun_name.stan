data {
  real logit; // causes error because logit is fun name
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
