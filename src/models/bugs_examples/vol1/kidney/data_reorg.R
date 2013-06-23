# turn the two-way array to one-way array. 
# so that the splitting could be easier. 
# 
# split the original data to censored data and right-censored data; 

source('kidney.old.data.R')

tv <- as.vector(t)
agev <- as.vector(age) 
t_cenv <- as.vector(t.cen) 

patient <- rep(1:N, 2)
sex <- c(sex, sex)
disease <- c(disease, disease) 


idx_rc <- which(is.na(tv)) 
idx_uc <- which(!is.na(tv))

t_uc <- tv[idx_uc] 
t_rc <- t_cenv[idx_rc]

age_uc <- agev[idx_uc]
age_rc <- agev[idx_rc] 

N_uc <- length(idx_uc)
N_rc <- length(idx_rc) 

patient_uc <- patient[idx_uc]
patient_rc <- patient[idx_rc]

disease_uc <- disease[idx_uc]
disease_rc <- disease[idx_rc]

sex_uc <- sex[idx_uc]
sex_rc <- sex[idx_rc]

# number of patients 
NP <- N; 

dump(c("NP", "N_uc", "N_rc", "t_uc", "t_rc", "age_uc", "age_rc", 
       "sex_uc", "sex_rc", 
       "patient_uc", "patient_rc", "disease_uc", "disease_rc"), 
     file = "kidney.data.R") 


