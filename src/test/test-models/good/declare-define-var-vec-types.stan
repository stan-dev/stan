data {
  vector[7] b0;
  row_vector[7] c0;
}
transformed data {
  vector[7] td_b1 = b0;
  row_vector[7] td_c1 = c0;
  vector[7] td_b2;
  {
    vector[7] loc_td_b;
    vector[7] loc_td_b1 = b0;
    row_vector[7] loc_td_c1 = c0;
  }
}
parameters {
  vector[7] par_b;
  row_vector[7] par_c;
}
transformed parameters {
  vector[7] tpar_b = par_b;
  row_vector[7] tpar_c = par_c;
  {
    vector[7] loc_tpar_b1 = b0;
    row_vector[7] loc_tpar_c1 = c0;
  }
}
model {
  row_vector[7] model_c1 = c0;
  {
    vector[7] loc_model_b1 = b0;
    row_vector[7] loc_model_c1 = c0;
  }
}
generated quantities {
  vector[7] gq_b1 = b0;
  row_vector[7] gq_c1 = c0;
  {
    vector[7] loc_gq_b1 = b0;
    row_vector[7] loc_gq_c1 = c0;
  }
}
