# split the original data to censored data and right-censored data 
source('mice.old.Rdata')

N_uncensored <- sum(is_censored == 0);
N_censored <- sum(is_censored == 1);

group_uncensored <- group[is_censored == 0];
group_censored <- group[is_censored == 1];

t_uncensored <- t[is_censored == 0];
censor_time <- 40

dump(c("N_uncensored", "N_censored", "M", "group_uncensored", "group_censored", "t_uncensored", "censor_time"), file = "mice.Rdata"); 
