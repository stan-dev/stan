data {
  real a;
  vector[7] b0;
  row_vector[7] c0;
}
transformed data {
  real b = a;
  vector[7] td_b1;
  vector[7] td_b2 = b0;
  row_vector[7] td_c1;
  row_vector[7] td_c2 = c0;
  {
    vector[7] loc_td_b;
    vector[7] loc_td_b1 = b0;
    row_vector[7] loc_td_c;
    row_vector[7] loc_td_c1 = c0;
  }
}
parameters {
  vector[7] par_b;
  vector[7] par_b1 = b0;
  row_vector[7] par_c;
  row_vector[7] par_c1 = c0;
}
transformed parameters {
  vector[7] tpar_b;
  vector[7] tpar_b1 = b0;
  row_vector[7] tpar_c;
  row_vector[7] tpar_c1 = c0;
  {
    vector[7] loc_tpar_b;
    vector[7] loc_tpar_b1 = b0;
    row_vector[7] loc_tpar_c;
    row_vector[7] loc_tpar_c1 = c0;
  }
}
model {
  vector[7] model_b;
  vector[7] model_b1 = b0;
  row_vector[7] model_c;
  row_vector[7] model_c1 = c0;
  {
    vector[7] loc_model_b;
    vector[7] loc_model_b1 = b0;
    row_vector[7] loc_model_c;
    row_vector[7] loc_model_c1 = c0;
  }
}
generated quantities {
  vector[7] gq_b;
  vector[7] gq_b1 = b0;
  row_vector[7] gq_c;
  row_vector[7] gq_c1 = c0;
  {
    vector[7] loc_gq_b;
    vector[7] loc_gq_b1 = b0;
    row_vector[7] loc_gq_c;
    row_vector[7] loc_gq_c1 = c0;
  }
}
