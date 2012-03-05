data {
  real(0,) x;
}
transformed data {
  real(0,1) y;
}
parameters {
  real(0,) sigma;
}
transformed parameters {
  real(0,) sigma_sq;
}
model {
   matrix(3,3) m1;
   matrix(3,3) m2;
   matrix(3,3) m3;
   real z;

   m3 <- m2';
   m3 <- -m1;
   m3 <- m1 + m2;
   m3 <- m1 - m2;
   m3 <- m1 * m2;
   m3 <- m1 * m2 + m3;
   m3 <- m1 + m2 * m3;
   m3 <- m1' * m2;
   m3 <- m1 * m2';
   m3 <- (m1 * m2)';
   z <- -z;
   z <- z + 1.0;
   z <- z - 2.0;
   z <- z * 3.0;
   z <- z / 4.0;
}
generated quantities {
  real(0,1) w;
}
