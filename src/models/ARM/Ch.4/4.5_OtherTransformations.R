library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ as.factor(mom_work)
data.list <- c("N", "kid_score", "mom_work")
kidscore_momwork.sf <- stan(file='kidscore_momwork.stan', data=data.list,
                            iter=1000, chains=4)
print(kidscore_momwork.sf)
