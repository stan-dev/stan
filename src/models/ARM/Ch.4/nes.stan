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

  vector[N] z_partyid7;      // standardization
  real partyid7_mean;
  real<lower=0> partyid7_sd;
  vector[N] z_real_ideo;
  real real_ideo_mean;
  real<lower=0> real_ideo_sd;
  vector[N] z_race_adj;
  real race_adj_mean;
  real<lower=0> race_adj_sd;
  vector[N] z_educ1;
  real educ1_mean;
  real<lower=0> educ1_sd;
  vector[N] z_gender;
  real gender_mean;
  real<lower=0> gender_sd;
  vector[N] z_income;
  real income_mean;
  real<lower=0> income_sd;
  vector[N] z_age30_44;
  real age30_44_mean;
  real<lower=0> age30_44_sd;
  vector[N] z_age45_64;
  real age45_64_mean;
  real<lower=0> age45_64_sd;
  vector[N] z_age65up;
  real age65up_mean;
  real<lower=0> age65up_sd;

  // age as factor
  for (n in 1:N) {
    age30_44[n] <- age_discrete[n] == 2;
    age45_64[n] <- age_discrete[n] == 3;
    age65up[n]  <- age_discrete[n] == 4;
  }

  // standardization
  partyid7_mean  <- mean(partyid7);
  partyid7_sd    <- sd(partyid7);
  z_partyid7     <- (partyid7 - partyid7_mean) / partyid7_sd;
  real_ideo_mean <- mean(real_ideo);
  real_ideo_sd   <- sd(real_ideo);
  z_real_ideo    <- (real_ideo - real_ideo_mean) / real_ideo_sd;
  race_adj_mean  <- mean(race_adj);
  race_adj_sd    <- sd(race_adj);
  z_race_adj     <- (race_adj - race_adj_mean) / race_adj_sd;
  educ1_mean     <- mean(educ1);
  educ1_sd       <- sd(educ1);
  z_educ1        <- (educ1 - educ1_mean) / educ1_sd;
  gender_mean    <- mean(gender);
  gender_sd      <- sd(gender);
  z_gender       <- (gender - gender_mean) / gender_sd;
  income_mean    <- mean(income);
  income_sd      <- sd(income);
  z_income       <- (income - income_mean) / income_sd;
  age30_44_mean  <- mean(age30_44);
  age30_44_sd    <- sd(age30_44);
  z_age30_44     <- (age30_44 - age30_44_mean) / age30_44_sd;
  age45_64_mean  <- mean(age45_64);
  age45_64_sd    <- sd(age45_64);
  z_age45_64     <- (age45_64 - age45_64_mean) / age45_64_sd;
  age65up_mean   <- mean(age65up);
  age65up_sd     <- sd(age65up);
  z_age65up      <- (age65up - age65up_mean) / age65up_sd;
}
parameters {
  vector[9] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_partyid7 ~ normal(z_beta[1] + z_beta[2] * z_real_ideo + z_beta[3] * z_race_adj 
                    + z_beta[4] * z_age30_44 + z_beta[5] * z_age45_64
                    + z_beta[6] * z_age65up + z_beta[7] * z_educ1
                    + z_beta[8] * z_gender + z_beta[9] * z_income,
                    z_sigma);
}
generated quantities {       // recovered parameter values
  vector[9] beta;
  real<lower=0> sigma;
  beta[1] <- partyid7_sd
             * (z_beta[1] - z_beta[2] * real_ideo_mean / real_ideo_sd
                - z_beta[3] * race_adj_mean / race_adj_sd
                - z_beta[4] * age30_44_mean / age30_44_sd
                - z_beta[5] * age45_64_mean / age45_64_sd
                - z_beta[6] * age65up_mean / age65up_sd
                - z_beta[7] * educ1_mean / educ1_sd
                - z_beta[8] * gender_mean / gender_sd
                - z_beta[9] * income_mean / income_sd)
             + partyid7_mean;
  beta[2] <- z_beta[2] * partyid7_sd / real_ideo_sd;
  beta[3] <- z_beta[3] * partyid7_sd / race_adj_sd;
  beta[4] <- z_beta[4] * partyid7_sd / age30_44_sd;
  beta[5] <- z_beta[5] * partyid7_sd / age45_64_sd;
  beta[6] <- z_beta[6] * partyid7_sd / age65up_sd;
  beta[7] <- z_beta[7] * partyid7_sd / educ1_sd;
  beta[8] <- z_beta[8] * partyid7_sd / gender_sd;
  beta[9] <- z_beta[9] * partyid7_sd / income_sd;
  sigma   <- partyid7_sd * z_sigma;
}
               
