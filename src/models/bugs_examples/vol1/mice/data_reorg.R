# split the original data to censored data and right-censored data 

source('mice.Rdata')


idx_uc <- which(is_censored == 0)  
idx_rc <- which(is_censored == 1)  

# uncensored group 
group_uc <- group[idx_uc] 
group_rc <- group[idx_rc] 

#
last_t_rc <- last_t[idx_rc] 
t_uc <- t[idx_uc] 

N_uc <- length(idx_uc) 
N_rc <- length(idx_rc)

dump(c("group_uc", "group_rc", "N_uc", "N_rc", "M", "t_uc", "last_t_rc"), file = "mice2.Rdata"); 


