data {
  vector[4] x;
}
parameters {
  vector[4] theta;
}
model {
  x ~ foo_whatev(theta);
}
