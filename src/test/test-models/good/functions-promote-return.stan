functions {
  vector foo(vector x, real y) {
    if (y < 10)
      return log(x);  // problem is here if x data and y parameter
    else
      return log(y * x);  // this promotes OK given x and y scalars
  }
}
transformed data {
  vector[3] x;
  x <- rep_vector(0,3);
}
parameters {
  real y;
}
model {
  foo(x,y) ~ normal(0,1);  // foo(x,y) has x double, y var, and return type var
}
