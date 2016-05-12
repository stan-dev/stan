data {
  real a;
  vector[3] b;
  real c[7];
  real d[8, 9];
}
parameters {
  real e;
  vector[3] f;
  real g[7];
  real h[8, 9];
}
model {
  //  increment_log_prob(-e^2 / 2);
  increment_log_prob(a);
  increment_log_prob(b);
  increment_log_prob(b);
  increment_log_prob(c);
  increment_log_prob(d);
  increment_log_prob(e);
  increment_log_prob(f);
  increment_log_prob(g);
  increment_log_prob(h);
}
