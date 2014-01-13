data {
  int<lower=0>m1;  // make sure didn't mess up spacing
  int <lower=0>m2;
  int<lower=0> m3;
  int <lower=0> m4;
}
transformed data {
  int intercept;
  intercept <- 5;  // failed in 1.0.2
}
model {

}
