library(rstan)
library(ggplot2)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq

if (!exists("kidiq_multi_preds.sm")) {
    if (file.exists("kidiq_multi_preds.sm.RData")) {
        load("kidiq_multi_preds.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_multi_preds.stan", model_name = "kidiq_multi_preds")
        kidiq_multi_preds.sm <- stan_model(stanc_ret = rt)
        save(kidiq_multi_preds.sm, file = "kidiq_multi_preds.sm.RData")
    }
}

data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_multi_preds.sf <- sampling(kidiq_multi_preds.sm, data.list)
print(kidiq_multi_preds.sf, pars = c("beta", "sigma", "lp__"))

# Figure 3.3
beta.post <- extract(kidiq_multi_preds.sf, "beta")$beta
beta.mean <- colMeans(beta.post)
kidiq.data <- data.frame(kid_score, mom_hs = as.factor(mom_hs), mom_iq)
levels(kidiq.data$mom_hs) <- c("No", "Yes")

p <- ggplot(kidiq.data, aes(x = mom_iq, y = kid_score, color = mom_hs)) +
     geom_point() +
     geom_abline(aes(intercept = beta.mean[1] + beta.mean[2] * (mom_hs == "Yes"),
                     slope = beta.mean[3], color = mom_hs)) +
     scale_x_continuous("Mother IQ score", breaks = c(80, 100, 120, 140)) +
     scale_y_continuous("Child test score", breaks = c(20, 60, 100, 140)) +
     scale_color_manual("Mother\ncompleted\nhigh\nschool",
                        values = c("No" = "black", "Yes" = "gray")) +
     theme_bw()
print(p)
