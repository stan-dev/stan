# turn the two-way array to one-way array. 
# so that the splitting could be easier. 
# 
# and we create an indices two-way array 
# to remember the original locations. 
# 
# split the original data to censored data and right-censored data; 

source('kidney.Rdata')

ijindex <- rbind(cbind(1:N, rep(1, N)),
                 cbind(1:N, rep(2, N))) 

tv <- as.vector(t)
agev <- as.vector(age) 
t_cenv <- as.vector(t.cen) 


idx_rc <- which(is.na(tv)) 
idx_uc <- which(!is.na(tv))

t_uc <- tv[idx_uc] 
t_rc <- t_cenv[idx_rc]

age_uc <- agev[idx_uc]
age_rc <- agev[idx_rc] 

N_uc <- length(idx_uc)
N_rc <- length(idx_rc) 

ijindex_uc <- ijindex[idx_uc, ]
ijindex_rc <- ijindex[idx_rc, ]

# number of patients 
NP <- N; 

dump(c("NP", "N_uc", "N_rc", "sex", "t_uc", "t_rc", "age_uc", "age_rc", 
       "disease", "ijindex_uc", "ijindex_rc"), 
     file = "kidney2.Rdata") 


