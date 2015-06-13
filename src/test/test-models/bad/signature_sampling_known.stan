data {
  vector[4] x;
}
parameters {
  vector[4] theta;
}
model {
  x ~ bernoulli_logit(theta);
}
