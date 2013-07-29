library(rstan)
library(ggplot2)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_iq

if (!exists("kidscore_momiq.sm")) {
    if (file.exists("kidscore_momiq.sm.RData")) {
        load("kidscore_momiq.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidscore_momiq.stan", model_name = "kidscore_momiq")
        kidscore_momiq.sm <- stan_model(stanc_ret = rt)
        save(kidscore_momiq.sm, file = "kidscore_momiq.sm.RData")
    }
}

data.list.2 <- c("N", "kid_score", "mom_iq")
kidscore_momiq.sf <- sampling(kidscore_momiq.sm, data.list.2)
print(kidscore_momiq.sf, pars = c("beta", "sigma", "lp__"))

### Figure 3.12

beta.post <- extract(kidscore_momiq.sf, "beta")$beta
beta.mean <- colMeans(beta.post)
resid <- kid_score - (beta.mean[1] + beta.mean[2] * mom_iq)
resid.sd <- sd(resid)

p <- ggplot(data.frame(mom_iq, resid), aes(x = mom_iq, y = resid)) +
    geom_point() +
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = c(-resid.sd, resid.sd), linetype = "dashed") +
    scale_x_continuous("Mother IQ score", breaks = seq(80, 140, 20)) +
    scale_y_continuous("Residuals", breaks = seq(-60, 40, 20)) 
print(p)
