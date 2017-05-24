transformed data {
  print("In transformed data");
  print("quitting time");
  reject("QUIT");
}
parameters {
  real y;
} 
transformed parameters {
  print("In transformed parameters");
}
model {
  print("In model block.");
  y ~ normal(0,1);
}
generated quantities {
  print("In generated quantities");
}
