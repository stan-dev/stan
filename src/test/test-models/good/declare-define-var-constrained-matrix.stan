data {
  int d;

  corr_matrix [d] d_corr_matrix;
  corr_matrix [d] d_corr_matrix_ar[1];

  cov_matrix [d] d_cov_matrix;
  cov_matrix [d] d_cov_matrix_ar[1];

  cholesky_factor_corr [d] d_cholesky_factor_corr;
  cholesky_factor_corr [d] d_cholesky_factor_corr_ar[1];

  cholesky_factor_cov [d] d_cholesky_factor_cov;
  cholesky_factor_cov [d] d_cholesky_factor_cov_ar[1];
}
transformed data {
  corr_matrix[d] td_corr_matrix1 = d_corr_matrix;
  corr_matrix[d] td_corr_matrix2 = d_corr_matrix_ar[1];
  corr_matrix[d] td_corr_matrix_ar[1] = d_corr_matrix_ar;

  cov_matrix[d] td_cov_matrix1 = d_cov_matrix;
  cov_matrix[d] td_cov_matrix2 = d_cov_matrix_ar[1];
  cov_matrix[d] td_cov_matrix_ar[1] = d_cov_matrix_ar;

  cholesky_factor_corr[d] td_cholesky_factor_corr1 = d_cholesky_factor_corr;
  cholesky_factor_corr[d] td_cholesky_factor_corr2 = d_cholesky_factor_corr_ar[1];
  cholesky_factor_corr[d] td_cholesky_factor_corr_ar[1] = d_cholesky_factor_corr_ar;

  cholesky_factor_cov[d] td_cholesky_factor_cov1 = d_cholesky_factor_cov;
  cholesky_factor_cov[d] td_cholesky_factor_cov2 = d_cholesky_factor_cov_ar[1];
  cholesky_factor_cov[d] td_cholesky_factor_cov_ar[1] = d_cholesky_factor_cov_ar;

  print("td_corr_matrix1 = ", td_corr_matrix1);
  print("td_corr_matrix2 = ", td_corr_matrix2);
  print("td_corr_matrix_ar = ", td_corr_matrix_ar);

  print("td_cov_matrix1 = ", td_cov_matrix1);
  print("td_cov_matrix2 = ", td_cov_matrix2);
  print("td_cov_matrix_ar = ", td_cov_matrix_ar);

  print("td_cholesky_factor_corr1 = ", td_cholesky_factor_corr1);
  print("td_cholesky_factor_corr2 = ", td_cholesky_factor_corr2);
  print("td_cholesky_factor_corr_ar = ", td_cholesky_factor_corr_ar);

  print("td_cholesky_factor_cov1 = ", td_cholesky_factor_cov1);
  print("td_cholesky_factor_cov2 = ", td_cholesky_factor_cov2);
  print("td_cholesky_factor_cov_ar = ", td_cholesky_factor_cov_ar);
}
transformed parameters {
  corr_matrix[d] tp_corr_matrix1 = d_corr_matrix;
  corr_matrix[d] tp_corr_matrix2 = d_corr_matrix_ar[1];
  corr_matrix[d] tp_corr_matrix_ar[1] = d_corr_matrix_ar;

  cov_matrix[d] tp_cov_matrix1 = d_cov_matrix;
  cov_matrix[d] tp_cov_matrix2 = d_cov_matrix_ar[1];
  cov_matrix[d] tp_cov_matrix_ar[1] = d_cov_matrix_ar;

  cholesky_factor_corr[d] tp_cholesky_factor_corr1 = d_cholesky_factor_corr;
  cholesky_factor_corr[d] tp_cholesky_factor_corr2 = d_cholesky_factor_corr_ar[1];
  cholesky_factor_corr[d] tp_cholesky_factor_corr_ar[1] = d_cholesky_factor_corr_ar;

  cholesky_factor_cov[d] tp_cholesky_factor_cov1 = d_cholesky_factor_cov;
  cholesky_factor_cov[d] tp_cholesky_factor_cov2 = d_cholesky_factor_cov_ar[1];
  cholesky_factor_cov[d] tp_cholesky_factor_cov_ar[1] = d_cholesky_factor_cov_ar;

  print("tp_corr_matrix1 = ", tp_corr_matrix1);
  print("tp_corr_matrix2 = ", tp_corr_matrix2);
  print("tp_corr_matrix_ar = ", tp_corr_matrix_ar);

  print("tp_cov_matrix1 = ", tp_cov_matrix1);
  print("tp_cov_matrix2 = ", tp_cov_matrix2);
  print("tp_cov_matrix_ar = ", tp_cov_matrix_ar);

  print("tp_cholesky_factor_corr1 = ", tp_cholesky_factor_corr1);
  print("tp_cholesky_factor_corr2 = ", tp_cholesky_factor_corr2);
  print("tp_cholesky_factor_corr_ar = ", tp_cholesky_factor_corr_ar);

  print("tp_cholesky_factor_cov1 = ", tp_cholesky_factor_cov1);
  print("tp_cholesky_factor_cov2 = ", tp_cholesky_factor_cov2);
  print("tp_cholesky_factor_cov_ar = ", tp_cholesky_factor_cov_ar);
}
model {
}
generated quantities {
  corr_matrix[d] gq_corr_matrix1 = d_corr_matrix;
  corr_matrix[d] gq_corr_matrix2 = d_corr_matrix_ar[1];
  corr_matrix[d] gq_corr_matrix_ar[1] = d_corr_matrix_ar;

  cov_matrix[d] gq_cov_matrix1 = d_cov_matrix;
  cov_matrix[d] gq_cov_matrix2 = d_cov_matrix_ar[1];
  cov_matrix[d] gq_cov_matrix_ar[1] = d_cov_matrix_ar;

  cholesky_factor_corr[d] gq_cholesky_factor_corr1 = d_cholesky_factor_corr;
  cholesky_factor_corr[d] gq_cholesky_factor_corr2 = d_cholesky_factor_corr_ar[1];
  cholesky_factor_corr[d] gq_cholesky_factor_corr_ar[1] = d_cholesky_factor_corr_ar;

  cholesky_factor_cov[d] gq_cholesky_factor_cov1 = d_cholesky_factor_cov;
  cholesky_factor_cov[d] gq_cholesky_factor_cov2 = d_cholesky_factor_cov_ar[1];
  cholesky_factor_cov[d] gq_cholesky_factor_cov_ar[1] = d_cholesky_factor_cov_ar;

  print("gq_corr_matrix1 = ", gq_corr_matrix1);
  print("gq_corr_matrix2 = ", gq_corr_matrix2);
  print("gq_corr_matrix_ar = ", gq_corr_matrix_ar);

  print("gq_cov_matrix1 = ", gq_cov_matrix1);
  print("gq_cov_matrix2 = ", gq_cov_matrix2);
  print("gq_cov_matrix_ar = ", gq_cov_matrix_ar);

  print("gq_cholesky_factor_corr1 = ", gq_cholesky_factor_corr1);
  print("gq_cholesky_factor_corr2 = ", gq_cholesky_factor_corr2);
  print("gq_cholesky_factor_corr_ar = ", gq_cholesky_factor_corr_ar);

  print("gq_cholesky_factor_cov1 = ", gq_cholesky_factor_cov1);
  print("gq_cholesky_factor_cov2 = ", gq_cholesky_factor_cov2);
  print("gq_cholesky_factor_cov_ar = ", gq_cholesky_factor_cov_ar);
}
