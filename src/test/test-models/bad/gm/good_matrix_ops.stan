transformed data {
  matrix[5,5] A;
  matrix[5,5] B;
  matrix[5,5] C;
  vector[5] b;
  vector[5] x;

  row_vector[5] c;
  row_vector[5] y;

  C <- A / B;
  C <- B \ A;

  x <- A \ b;
  
  y <- c / A;

  A <- A ./ A;

  A <- A .* A;

  x <- x ./ x;
  x <- x .* x;

  y <- y ./ y;
  y <- y .* y;

}
model {
}