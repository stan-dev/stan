parameters {
  array[4, 5] matrix<lower=0, upper=1>[2, 3] p_ar_mat;
}
model {
  for (i in 1 : 4) 
    for (j in 1 : 5) 
      for (k in 1 : 2) 
        for (l in 1 : 3) 
          p_ar_mat[i, j, k, l] ~ normal(0, 1);
}
generated quantities {
  array[4, 5] matrix[2, 3] gq_ar_mat;
  for (i in 1 : 4) 
    for (j in 1 : 5) 
      for (k in 1 : 2) 
        for (l in 1 : 3) 
          gq_ar_mat[i, j, k, l] = p_ar_mat[i, j, k, l];
}

