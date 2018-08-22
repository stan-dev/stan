// programs to tickle bug 2612
// sampling stmts w/ function calls

transformed data {
  matrix[2,2] M = rep_matrix(0, 2, 2);
}
model {
  to_vector(M) ~ normal(0, 1);
}
