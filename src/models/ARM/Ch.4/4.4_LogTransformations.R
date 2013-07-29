library(rstan)
library(ggplot2)

### Data

source("earnings.data.R", echo = TRUE)

### Log transformation

if (!exists("logearn_height.sm")) {
    if (file.exists("logearn_height.sm.RData")) {
        load("logearn_height.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("logearn_height.stan", model_name = "logearn_height")
        logearn_height.sm <- stan_model(stanc_ret = rt)
        save(logearn_height.sm, file = "logearn_height.sm.RData")
    }
}

data.list.1 <- c("N", "earn", "height")
logearn_height.sf <- sampling(logearn_height.sm, data.list.1)
print(logearn_height.sf, pars = c("beta", "sigma", "lp__"))

# Figure 4.3

beta.post.1 <- extract(logearn_height.sf, "beta")$beta
beta.mean.1 <- colMeans(beta.post.1)
n <- 20
ndx <- sample(nrow(beta.post.1), n)
# left
earnings.post.1 <- data.frame(sampled.int = beta.post.1[ndx, 1],
                              sampled.slope = beta.post.1[ndx, 2],
                              id = ndx)
p1 <- ggplot(data.frame(height, logearn = log(earn)), aes(x = height, y = logearn)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.2)) +
    geom_abline(aes(intercept = sampled.int, slope = sampled.slope, group = id),
                data = earnings.post.1, alpha = 0.1) +
    geom_abline(aes(intercept = beta.mean.1[1], slope = beta.mean.1[2])) +
    scale_x_continuous("height", breaks = seq(60, 75, 5)) +
    scale_y_continuous("log(earnings)", breaks = seq(6, 12, 2)) +
    ggtitle("Log regressione plotted on log scale") +
    theme_bw()
print(p1)

# right
dev.new()
earnings.post.2 <- data.frame(c(), c(), c()) # empty data frame
len <- length(earn)
for (i in ndx) {
    earn.disp <- exp(beta.post.1[i,1] + beta.post.1[i,2] * height)
    earnings.post.2 <- rbind(earnings.post.2,
                             data.frame(id = rep(i, len), height, earn.disp))
}
p2 <- ggplot(data.frame(height, earn), aes(x = height, y = earn)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.2)) +
    geom_line(aes(x = height, y = earn.disp, group = id),
              data = earnings.post.2, alpha = 0.1) +
    geom_line(aes(x = height, y = exp(beta.mean.1[1] + beta.mean.1[2] * height))) +
    scale_x_continuous("height", breaks = seq(60, 75, 5)) +
    scale_y_continuous("earnings", breaks = c(0, 100000, 200000),
                       labels = c("0", "100000", "200000")) +
    theme(axis.text.y = element_text(angle = 90, hjust = 0.5)) +
    ggtitle("Log regression plotted on original scale") +
    theme_bw()  
print(p2)

### Log-base-10 transformation

if (!exists("log10earn_height.sm")) {
    if (file.exists("log10earn_height.sm.RData")) {
        load("log10earn_height.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("log10earn_height.stan", model_name = "log10earn_height")
        log10earn_height.sm <- stan_model(stanc_ret = rt)
        save(log10earn_height.sm, file = "log10earn_height.sm.RData")
    }
}

log10earn_height.sf <- sampling(log10earn_height.sm, data.list.1)
print(log10earn_height.sf, pars = c("beta", "sigma", "lp__"))

### Log scale regression model

if (!exists("logearn_height_male.sm")) {
    if (file.exists("logearn_height_male.sm.RData")) {
        load("logearn_height_male.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("logearn_height_male.stan", model_name = "logearn_height_male")
        logearn_height_male.sm <- stan_model(stanc_ret = rt)
        save(logearn_height_male.sm, file = "logearn_height_male.sm.RData")
    }
}

data.list.2 <- c("N", "earn", "height", "male")
logearn_height_male.sf <- sampling(logearn_height_male.sm, data.list.2)
print(logearn_height_male.sf, pars = c("beta", "sigma", "lp__"))

### Including interactions

if (!exists("logearn_interaction.sm")) { if
    (file.exists("logearn_interaction.sm.RData")) {
    load("logearn_interaction.sm.RData", verbose = TRUE) } else { rt <-
    stanc("logearn_interaction.stan", model_name = "logearn_interaction")
    logearn_interaction.sm <- stan_model(stanc_ret = rt)
    save(logearn_interaction.sm, file = "logearn_interaction.sm.RData") } }

logearn_interaction.sf <- sampling(logearn_interaction.sm, data.list.2)
print(logearn_interaction.sf, pars = c("beta", "sigma", "lp__"))

### Linear transformations

if (!exists("logearn_interaction_z.sm")) {
    if (file.exists("logearn_interaction_z.sm.RData")) {
        load("logearn_interaction_z.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("logearn_interaction_z.stan", model_name = "logearn_interaction_z")
        logearn_interaction_z.sm <- stan_model(stanc_ret = rt)
        save(logearn_interaction_z.sm, file = "logearn_interaction_z.sm.RData")
    }
}

logearn_interaction_z.sf <- sampling(logearn_interaction_z.sm, data.list.2)
print(logearn_interaction_z.sf, pars = c("beta", "sigma", "lp__"))

### Log-log model

if (!exists("logearn_logheight.sm")) {
    if (file.exists("logearn_logheight.sm.RData")) {
        load("logearn_logheight.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("logearn_logheight.stan", model_name = "logearn_logheight")
        logearn_logheight.sm <- stan_model(stanc_ret = rt)
        save(logearn_logheight.sm, file = "logearn_logheight.sm.RData")
    }
}

logearn_logheight.sf <- sampling(logearn_logheight.sm, data.list.2)
print(logearn_logheight.sf, pars = c("beta", "sigma", "lp__"))
