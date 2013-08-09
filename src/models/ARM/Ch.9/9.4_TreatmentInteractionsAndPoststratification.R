library(rstan)
library(ggplot2)

### Data

source("electric_grade4.data.R", echo = TRUE)

### Model with only treatment indicator: post_test ~ treatment

if (!exists("electric_tr.sm")) {
    if (file.exists("electric_tr.sm.RData")) {
        load("electric_tr.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_tr.stan", model_name = "electric_tr")
        electric_tr.sm <- stan_model(stanc_ret = rt)
        save(electric_tr.sm, file = "electric_tr.sm.RData")
    }
}

data.list <- c("N", "post_test", "pre_test", "treatment")
electric_tr.sf <- sampling(electric_tr.sm, data.list)
print(electric_tr.sf)

### Model controlling for pre-test: post_test ~ treatment + pre_test

if (!exists("electric_trpre.sm")) {
    if (file.exists("electric_trpre.sm.RData")) {
        load("electric_trpre.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_trpre.stan", model_name = "electric_trpre")
        electric_trpre.sm <- stan_model(stanc_ret = rt)
        save(electric_trpre.sm, file = "electric_trpre.sm.RData")
    }
}

electric_trpre.sf <- sampling(electric_trpre.sm, data.list)
print(electric_trpre.sf)

### Model with interaction: post_test ~ pre_test + treatment + pre_test:treatment

if (!exists("electric_inter.sm")) {
    if (file.exists("electric_inter.sm.RData")) {
        load("electric_inter.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("electric_inter.stan", model_name = "electric_inter")
        electric_inter.sm <- stan_model(stanc_ret = rt)
        save(electric_inter.sm, file = "electric_inter.sm.RData")
    }
}

electric_inter.sf <- sampling(electric_inter.sm, data.list)
print(electric_inter.sf, pars = c("beta", "lp__"))

## Figure 9.7

inter.ggdf <- data.frame(c(), c(), c(), c(), c(), c(), c(), c()) # empty data frame
for (i in 1:4) {
    source(paste("electric_grade", i, ".data.R", sep = ""))
    temp <- data.frame(post_test, pre_test, grade, treatment)
    data.list <- c("N", "post_test", "pre_test", "treatment")
    sf <- sampling(electric_inter.sm, data.list)
    beta.post <- extract(sf, "beta")$beta
    beta.mean <- colMeans(beta.post)
    temp$beta1 <- beta.mean[1]
    temp$beta2 <- beta.mean[2]
    temp$beta3 <- beta.mean[3]
    temp$beta4 <- beta.mean[4]
    inter.ggdf <- rbind(inter.ggdf, temp)
}
inter.ggdf$grade <- factor(inter.ggdf$grade)
levels(inter.ggdf$grade) <- paste("Grade", levels(inter.ggdf$grade))
inter.ggdf$treatment <- factor(inter.ggdf$treatment)

p1 <- ggplot(inter.ggdf, aes(x = pre_test, y = post_test)) +
    geom_point(aes(shape = treatment)) +
    scale_shape_manual(values = c(16, 1)) +
    geom_abline(aes(intercept = beta1 + beta2 * (as.numeric(treatment)-1),
                    slope = beta3 + beta4 * (as.numeric(treatment)-1),
                    linetype = treatment)) +
    scale_linetype_manual(values = c(2, 1)) +
    facet_grid(. ~ grade) +
    scale_x_continuous(expression(paste("pre-test, ", x[i])),
                       limits = c(0, 125)) +
    scale_y_continuous(expression(paste("post-test, ", y[i])),
                       limits = c(0, 125)) +
    theme(legend.position = "none")
print(p1)

## Uncertainty (Figure 9.8)

source("electric_grade4.data.R")
beta.post <- extract(electric_inter.sf, "beta")$beta
beta.mean <- colMeans(beta.post)
n <- 20
ndx <- sample(nrow(beta.post), n)
uncertainty.ggdf <- data.frame(sampled_int = beta.post[ndx, 2],
                               sampled_slope = beta.post[ndx, 4],
                               id = ndx)
dev.new()
p2 <- ggplot(data.frame(pre_test),
             aes(x = pre_test, y = beta.mean[2] + beta.mean[4] * pre_test)) +
    geom_line(size = 0) +
    geom_abline(aes(intercept = beta.mean[2], slope = beta.mean[4])) +
    geom_abline(aes(intercept = sampled_int, slope = sampled_slope, group = id),
                data = uncertainty.ggdf, alpha = 0.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_x_continuous("pre-test") +
    scale_y_continuous("treatment effect", limits = c(-5, 10)) +
    ggtitle("treatment effect in grade 4")
print(p2)

## Compute the average treatment effect & summarize

n.iter <- nrow(beta.post)
effect <- array(NA, c(n.iter, N))
for (i in 1:n.iter)
    effect[i,] <- beta.post[i,2] + beta.post[i,4] * pre_test
avg.effect <- rowMeans(effect)
print(c(mean(avg.effect), sd(avg.effect)))
