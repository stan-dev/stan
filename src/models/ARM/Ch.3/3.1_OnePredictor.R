library(rstan)
library(ggplot2)

### Data

source("kidiq.data.R", echo = TRUE)

### First model: kid_score ~ mom_hs

if (!exists("kidscore_momhs.sm")) {
    if (file.exists("kidscore_momhs.sm.RData")) {
        load("kidscore_momhs.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidscore_momhs.stan", model_name = "kidscore_momhs")
        kidscore_momhs.sm <- stan_model(stanc_ret = rt)
        save(kidscore_momhs.sm, file = "kidscore_momhs.sm.RData")
    }
}

data.list.1 <- c("N", "kid_score", "mom_hs")
kidscore_momhs.sf <- sampling(kidscore_momhs.sm, data.list.1)
print(kidscore_momhs.sf, pars = c("beta", "sigma", "lp__"))

### Second model: kid_score ~ mom_iq

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

# Figure 3.1
kidiq.data <- data.frame(kid_score, mom_hs, mom_iq)
beta.post.1 <- extract(kidscore_momhs.sf, "beta")$beta
beta.mean.1 <- colMeans(beta.post.1)
p1 <- ggplot(kidiq.data, aes(x = mom_hs, y = kid_score)) +
    geom_jitter(position = position_jitter(width = 0.05, height = 0)) +
    geom_abline(aes(intercept = beta.mean.1[1], slope = beta.mean.1[2])) +
    scale_x_continuous("Mother completed high school", breaks = c(0, 1)) +
    scale_y_continuous("Child test score", breaks = c(20, 60, 100, 140)) +
    theme_bw()
print(p1)

# Figure 3.2
dev.new()
beta.post.2 <- extract(kidscore_momiq.sf, "beta")$beta
beta.mean.2 <- colMeans(beta.post.2)
p2 <- ggplot(kidiq.data, aes(x = mom_iq, y = kid_score)) +
    geom_point() +
    geom_abline(aes(intercept = beta.mean.2[1], slope = beta.mean.2[2])) +
    scale_x_continuous("Mother IQ score", breaks = c(80, 100, 120, 140)) +
    scale_y_continuous("Child test score", breaks = c(20, 60, 100, 140)) +
    theme_bw()
print(p2)
