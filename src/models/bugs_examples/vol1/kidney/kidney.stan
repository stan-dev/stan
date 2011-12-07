data {
  int(0,) N_uc;
  int(0,) N_rc;
  int(0,) NP; 
  int sex[NP]; 
  int disease[NP];
  double(0,) t_uc[N_uc]; 
  double(0,) t_cen_rc[N_rc]; 
  int ijindex_rc[N_rc, 2]; 
  int ijindex_uc[N_uc, 2]; 
} 
parameters {
  double alpha; 
  double beta_sex;
  double beta_disease2; 
  double beta_disease3; 
  double beta_disease4; 
  double b[NP]; 
  double(0, ) r; 
  double(0, ) tau;
} 
derived parameters {
  double(0,) sigma;
  double yabeta_disease[4];
  yabeta_disease[1] <- 0; 
  yabeta_disease[2] <- beta_disease2;
  yabeta_disease[3] <- beta_disease3;
  yabeta_disease[4] <- beta_disease4;
  sigma <- sqrt(1 / tau); 
}
derived data {
  int tmp_i; 
  int tmp_j; 
}; 
model {

  alpha ~ normal(0.0, 100); 
  beta_age ~ normal(0.0, 100); 
  beta_sex ~ normal(0.0, 100);
  beta_disease2 ~ normal(0, 100); 
  beta_disease3 ~ normal(0, 100); 
  beta_disease4 ~ normal(0, 100); 

  tau ~ gamma(1.0E-3, 1.0E-3);
  r ~ gamma(1.0, 1.0E-3); 

  for (i in 1:NP) b[i] ~ normal(0.0, sigma);   
  for (i in 1:N_uc) {
    tmp_i <- ijindex_uc[i, 1];
    tmp_j <- ijindex_uc[i, 2];
    t_uc[i] ~ weibull(r, exp(-(alpha + beta_age * age_uc[i] + beta_sex * sex[tmp_j]  
                               + yabeta_disease[disease[tmp_i]] + b[tmp_i]) / r));
  } 
  for (i in 1:N_rc) {
    tmp_i <- ijindex_rc[i, 1];
    tmp_j <- ijindex_rc[i, 2];

    1 ~ bernoulli(exp(-pow(t_rc[i] / exp(-(alpha + beta_age * age_rc[i] + beta_sex * sex[tmp_j]
                                         + yabeta_disease[disease[tmp_i]] + b[tmp_i]) / r), r)));
  }
}
