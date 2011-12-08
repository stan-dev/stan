data {
  int(0,) NP; 
  int(0,) N_uc;
  int(0,) N_rc;
  double(0,) t_uc[N_uc]; 
  double(0,) t_rc[N_rc]; 
  int disease_uc[N_uc]; 
  int disease_rc[N_rc]; 
  int patient_uc[N_uc]; 
  int patient_rc[N_rc]; 
  int sex_uc[N_uc]; 
  int sex_rc[N_rc]; 
  int age_uc[N_uc]; 
  int age_rc[N_rc]; 
} 
parameters {
  double alpha; 
  double beta_age;
  double beta_sex;
  double beta_disease2; 
  double beta_disease3; 
  double beta_disease4; 
  double(0,) r; 
  double(0,) tau;
  double b[NP]; 
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

model {
  alpha ~ normal(0, 100); 
  beta_age ~ normal(0, 100); 
  beta_sex ~ normal(0, 100);
  beta_disease2 ~ normal(0, 100); 
  beta_disease3 ~ normal(0, 100); 
  beta_disease4 ~ normal(0, 100); 

  tau ~ gamma(1.0E-3, 1.0E-3);
  r ~ gamma(1, 1.0E-3); 

  for (i in 1:NP) b[i] ~ normal(0, sigma);   
  for (i in 1:N_uc) {
    t_uc[i] ~ weibull(r, exp(-(alpha + beta_age * age_uc[i] + beta_sex * sex_uc[i] 
                               + yabeta_disease[disease_uc[i]] + b[patient_uc[i]]) / r));
  } 
  for (i in 1:N_rc) {
    1 ~ bernoulli(exp(-pow(t_rc[i] / exp(-(alpha + beta_age * age_rc[i] + beta_sex * sex_rc[i] 
                                         + yabeta_disease[disease_rc[i]] + b[patient_rc[i]]) / r), r)));
    //TODO: try the weibull_p 
    // 0 ~ bernoulli(weibull_p(t_rc[i], exp(-(alpha + beta_age * age_rc[i] + beta_sex * sex_rc[i] 
    //                                      + yabeta_disease[disease_rc[i]] + b[patient_rc[i]]) / r), r));
  }
}
