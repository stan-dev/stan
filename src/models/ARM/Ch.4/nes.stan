data {
  int<lower=0> N;
  vector[N] partyid7;
  vector[N] real_ideo;
  vector[N] race_adj;
  vector[N] educ1;
  vector[N] gender;
  vector[N] income;
  int age_discrete[N];
}
transformed data {
  vector[N] age30_44;        // age as factor
  vector[N] age45_64;
  vector[N] age65up;

  for (n in 1:N) {
    age30_44[n] <- age_discrete[n] == 2;
    age45_64[n] <- age_discrete[n] == 3;
    age65up[n]  <- age_discrete[n] == 4;
  }
}
parameters {
  vector[9] beta;
  real<lower=0> sigma;
}
model {                      // vectorization
  partyid7 ~ normal(beta[1] + beta[2] * real_ideo + beta[3] * race_adj 
                    + beta[4] * age30_44 + beta[5] * age45_64
                    + beta[6] * age65up + beta[7] * educ1
                    + beta[8] * gender + beta[9] * income,
                    sigma);
}
