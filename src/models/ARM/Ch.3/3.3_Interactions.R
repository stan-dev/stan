library(rstan)
library(ggplot2)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq
data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_interaction <- stan(file='kidiq_interaction.stan', data=data.list,
                          iter=1000, chains=4)
print(kidiq_interaction, pars = c("beta", "sigma", "lp__"))

### Figures
beta.post <- extract(kidiq_interaction, "beta")$beta
beta.mean <- colMeans(beta.post)
kidiq.data <- data.frame(kid_score, mom_hs = as.factor(mom_hs), mom_iq)
levels(kidiq.data$mom_hs) <- c("No", "Yes")

# Figure 3.4 (a)
p <- ggplot(kidiq.data, aes(x = mom_iq, y = kid_score, color = mom_hs)) +
     geom_point() +
     geom_abline(aes(intercept = beta.mean[1] + beta.mean[2] * (mom_hs == "Yes"),
                     slope = beta.mean[3] + beta.mean[4] * (mom_hs == "Yes"),
                     color = mom_hs)) +
     scale_color_manual("Mother\ncompleted\nhigh\nschool",
                        values = c("No" = "black", "Yes" = "gray")) +
     theme_bw()
print(p +
      scale_x_continuous("Mother IQ score", breaks = seq(80, 140, 20)) +
      scale_y_continuous("Child test score", breaks = seq(20, 140, 40)))

# Figure 3.4 (b)
dev.new()
print(p +
      scale_x_continuous("Mother IQ score", limits = c(0, 150),
                         breaks = seq(0, 150, 50)) +
      scale_y_continuous("Child test score", limits = c(-15, 150),
                         breaks = c(0, 50, 100)))
