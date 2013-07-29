library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ as.factor(mom_work)

if (!exists("kidscore_momwork.sm")) {
    if (file.exists("kidscore_momwork.sm.RData")) {
        load("kidscore_momwork.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidscore_momwork.stan", model_name = "kidscore_momwork")
        kidscore_momwork.sm <- stan_model(stanc_ret = rt)
        save(kidscore_momwork.sm, file = "kidscore_momwork.sm.RData")
    }
}

data.list <- c("N", "kid_score", "mom_work")
kidscore_momwork.sf <- sampling(kidscore_momwork.sm, data.list)
print(kidscore_momwork.sf)
