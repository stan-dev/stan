parameters {
  real e;
  real pi;
  real log2;
  real log10;
  real sqrt2;
  real not_a_number;
  real positive_infinity;
  real negative_infinity;
  real machine_precision;
}
transformed parameters {
  real mu;
  mu = e() + pi() + log2() + log10() + sqrt2() + not_a_number()
    + positive_infinity() + negative_infinity() + machine_precision();
}
model {
}
