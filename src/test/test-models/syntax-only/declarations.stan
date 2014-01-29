data {
  int a0;
  int a1[1];
  int a2[2,3];

  real b0;
  real b1[4];
  real b2[5,6];
  
  vector[7] c0;
  vector[7] c1[8];
  vector[7] c2[8,9];

  row_vector[7] d0;
  row_vector[7] d1[8];
  row_vector[7] d2[8,9];

  matrix[8,9] e;
  matrix[8,9] e1[11];
  matrix[8,9] e2[11,12];

  simplex[8] f;
  simplex[8] f1[9];
  simplex[8] f2[9,10];
  
  ordered[9] g;
  ordered[9] g1[11];
  ordered[9] g2[12,13];

  positive_ordered[9] h;
  positive_ordered[9] h1[11];
  positive_ordered[9] h2[12,13];

  corr_matrix[10] j;
  corr_matrix[10] j1[3];
  corr_matrix[10] j2[5,6];

  cov_matrix[10] k;
  cov_matrix[10] k1[3];
  cov_matrix[10] k2[5,6];
}
transformed data {
  int td_a0;
  int td_a1[1];
  int td_a2[2,3]; 

  real td_b0;
  real td_b1[4];
  real td_b2[5,6];
  
  vector[7] td_c0;
  vector[7] td_c1[8];
  vector[7] td_c2[8,9];

  row_vector[7] td_d0;
  row_vector[7] td_d1[8];
  row_vector[7] td_d2[8,9];

  matrix[8,9] td_e;
  matrix[8,9] td_e1[11];
  matrix[8,9] td_e2[11,12];

  simplex[8] td_f;
  simplex[8] td_f1[9];
  simplex[8] td_f2[9,10];
  
  ordered[9] td_g;
  ordered[9] td_g1[11];
  ordered[9] td_g2[12,13];

  positive_ordered[9] td_h;
  positive_ordered[9] td_h1[11];
  positive_ordered[9] td_h2[12,13];

  corr_matrix[10] td_j;
  corr_matrix[10] td_j1[3];
  corr_matrix[10] td_j2[5,6];

  cov_matrix[10] td_k;
  cov_matrix[10] td_k1[3];
  cov_matrix[10] td_k2[5,6];

  {
    int loc_td_a0;
    int loc_td_a1[1];
    int loc_td_a2[2,3]; 

    real loc_td_b0;
    real loc_td_b1[4];
    real loc_td_b2[5,6];
  
    vector[7] loc_td_c0;
    vector[7] loc_td_c1[8];
    vector[7] loc_td_c2[8,9];

    row_vector[7] loc_td_d0;
    row_vector[7] loc_td_d1[8];
    row_vector[7] loc_td_d2[8,9];

    matrix[8,9] loc_td_e;
    matrix[8,9] loc_td_e1[11];
    matrix[8,9] loc_td_e2[11,12];
  }
}
parameters {
  real par_b0;
  real par_b1[4];
  real par_b2[5,6];
  
  vector[7] par_c0;
  vector[7] par_c1[8];
  vector[7] par_c2[8,9];

  row_vector[7] par_d0;
  row_vector[7] par_d1[8];
  row_vector[7] par_d2[8,9];

  matrix[8,9] par_e;
  matrix[8,9] par_e1[11];
  matrix[8,9] par_e2[11,12];

  simplex[8] par_f;
  simplex[8] par_f1[9];
  simplex[8] par_f2[9,10];
  
  ordered[9] par_g;
  ordered[9] par_g1[11];
  ordered[9] par_g2[12,13];

  positive_ordered[9] par_h;
  positive_ordered[9] par_h1[11];
  positive_ordered[9] par_h2[12,13];

  corr_matrix[10] par_j;
  corr_matrix[10] par_j1[3];
  corr_matrix[10] par_j2[5,6];

  cov_matrix[10] par_k;
  cov_matrix[10] par_k1[3];
  cov_matrix[10] par_k2[5,6];
}
transformed parameters {
  real tpar_b0;
  real tpar_b1[4];
  real tpar_b2[5,6];
  
  vector[7] tpar_c0;
  vector[7] tpar_c1[8];
  vector[7] tpar_c2[8,9];

  row_vector[7] tpar_d0;
  row_vector[7] tpar_d1[8];
  row_vector[7] tpar_d2[8,9];

  matrix[8,9] tpar_e;
  matrix[8,9] tpar_e1[11];
  matrix[8,9] tpar_e2[11,12];

  simplex[8] tpar_f;
  simplex[8] tpar_f1[9];
  simplex[8] tpar_f2[9,10];
  
  ordered[9] tpar_g;
  ordered[9] tpar_g1[11];
  ordered[9] tpar_g2[12,13];

  positive_ordered[9] tpar_h;
  positive_ordered[9] tpar_h1[11];
  positive_ordered[9] tpar_h2[12,13];

  corr_matrix[10] tpar_j;
  corr_matrix[10] tpar_j1[3];
  corr_matrix[10] tpar_j2[5,6];

  cov_matrix[10] tpar_k;
  cov_matrix[10] tpar_k1[3];
  cov_matrix[10] tpar_k2[5,6];

  {
    int loc_tpar_a0;
    int loc_tpar_a1[1];
    int loc_tpar_a2[2,3]; 

    real loc_tpar_b0;
    real loc_tpar_b1[4];
    real loc_tpar_b2[5,6];
  
    vector[7] loc_tpar_c0;
    vector[7] loc_tpar_c1[8];
    vector[7] loc_tpar_c2[8,9];

    row_vector[7] loc_tpar_d0;
    row_vector[7] loc_tpar_d1[8];
    row_vector[7] loc_tpar_d2[8,9];

    matrix[8,9] loc_tpar_e;
    matrix[8,9] loc_tpar_e1[11];
    matrix[8,9] loc_tpar_e2[11,12];
  }
}

model {

  {
    int loc_model_a0;
    int loc_model_a1[1];
    int loc_model_a2[2,3]; 

    real loc_model_b0;
    real loc_model_b1[4];
    real loc_model_b2[5,6];
  
    vector[7] loc_model_c0;
    vector[7] loc_model_c1[8];
    vector[7] loc_model_c2[8,9];

    row_vector[7] loc_model_d0;
    row_vector[7] loc_model_d1[8];
    row_vector[7] loc_model_d2[8,9];

    matrix[8,9] loc_model_e;
    matrix[8,9] loc_model_e1[11];
    matrix[8,9] loc_model_e2[11,12];
  }

}
generated quantities {
  real gq_b0;
  real gq_b1[4];
  real gq_b2[5,6];
  
  vector[7] gq_c0;
  vector[7] gq_c1[8];
  vector[7] gq_c2[8,9];

  row_vector[7] gq_d0;
  row_vector[7] gq_d1[8];
  row_vector[7] gq_d2[8,9];

  matrix[8,9] gq_e;
  matrix[8,9] gq_e1[11];
  matrix[8,9] gq_e2[11,12];

  simplex[8] gq_f;
  simplex[8] gq_f1[9];
  simplex[8] gq_f2[9,10];
  
  ordered[9] gq_g;
  ordered[9] gq_g1[11];
  ordered[9] gq_g2[12,13];

  positive_ordered[9] gq_h;
  positive_ordered[9] gq_h1[11];
  positive_ordered[9] gq_h2[12,13];

  corr_matrix[10] gq_j;
  corr_matrix[10] gq_j1[3];
  corr_matrix[10] gq_j2[5,6];

  cov_matrix[10] gq_k;
  cov_matrix[10] gq_k1[3];
  cov_matrix[10] gq_k2[5,6];

  {
    int loc_gq_a0;
    int loc_gq_a1[1];
    int loc_gq_a2[2,3]; 

    real loc_gq_b0;
    real loc_gq_b1[4];
    real loc_gq_b2[5,6];
  
    vector[7] loc_gq_c0;
    vector[7] loc_gq_c1[8];
    vector[7] loc_gq_c2[8,9];

    row_vector[7] loc_gq_d0;
    row_vector[7] loc_gq_d1[8];
    row_vector[7] loc_gq_d2[8,9];

    matrix[8,9] loc_gq_e;
    matrix[8,9] loc_gq_e1[11];
    matrix[8,9] loc_gq_e2[11,12];
  }
}
