model {
  real foo;
  foo = 1;
  target += 0;
  foo = target();
  foo = lmultiply(1, 1);
  foo = lchoose(1, 1);
  foo = normal_lpdf(0.5| 0, 1);
  foo = normal_lcdf(0.5| 0, 1);
  foo = normal_lccdf(0.5| 0, 1);
}

