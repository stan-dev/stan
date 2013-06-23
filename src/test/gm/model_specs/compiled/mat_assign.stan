transformed data {
  matrix[2,2] X;
  row_vector[2] y;
  y[1] <- 10;
  y[2] <- 100;

  X[1] <- y;
  X[2] <- y;
  print("X=",X);
}
parameters {
  real z;
}
transformed parameters {
  matrix[2,2] Xvar;
  matrix[2,2] Xvar2;
  row_vector[2] yvar;
  
  yvar[1] <- 15.9;
  yvar[2] <- 42.7;

  Xvar[1] <- y;
  Xvar[2] <- y;

  Xvar2[1] <- yvar;
  Xvar2[2] <- yvar;
}
model {
  z ~ normal(0,1);
}
