functions {
  real unit_normal_log(real y) {
    return normal_log(y,0,1); // print's messing this up now
  } 
}
parameters {
  real y;
}
model {
  y ~ unit_normal();
}
