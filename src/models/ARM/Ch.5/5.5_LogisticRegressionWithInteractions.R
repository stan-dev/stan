library(rstan)
library(ggplot2)

### Data

source("wells.data.R", echo = TRUE)

### Logistic regression with interactions

## Model: switched ~ dist/100 + arsenic + dist/100:arsenic

if (!exists("wells_interaction.sm")) {
    if (file.exists("wells_interaction.sm.RData")) {
        load("wells_interaction.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_interaction.stan", model_name = "wells_interaction")
        wells_interaction.sm <- stan_model(stanc_ret = rt)
        save(wells_interaction.sm, file = "wells_interaction.sm.RData")
    }
}

data.list.1 <- c("N", "switched", "dist", "arsenic")
wells_interaction.sf <- sampling(wells_interaction.sm, data.list.1)
print(wells_interaction.sf, pars = c("beta", "lp__"))

### Centering the input variabiles

## Model: switched ~ c_dist100 + c_arsenic + c_dist100:c_arsenic
## c_dist100 <- dist/100 - mean(dist/100)
## c_arsenic <- arsenic - mean(arsenic)

if (!exists("wells_interaction_c.sm")) {
    if (file.exists("wells_interaction_c.sm.RData")) {
        load("wells_interaction_c.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_interaction_c.stan", model_name = "wells_interaction_c")
        wells_interaction_c.sm <- stan_model(stanc_ret = rt)
        save(wells_interaction_c.sm, file = "wells_interaction_c.sm.RData")
    }
}

wells_interaction_c.sf <- sampling(wells_interaction_c.sm, data.list.1)
print(wells_interaction_c.sf)

## Figure 5.12: graphing the (first) model with interaction

beta.post.1 <- extract(wells_interaction.sf, "beta")$beta
beta.mean.1 <- colMeans(beta.post.1)

# left
p1 <- ggplot(data.frame(switched, dist), aes(dist, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- cbind(1, x/100, 0.5, 0.5*x/100) %*% beta.mean.1))) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- cbind(1, x/100, 1.0, 1.0*x/100) %*% beta.mean.1))) +
    annotate("text", x = c(50,75), y = c(0.35, 0.55),
             label = c("if As = 0.5", "if As = 1.0"), size = 4) +
    scale_x_continuous("Distance (in meters) to the nearest safe well",
                       breaks = seq(from = 0, by = 50, length.out = 7)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
plot(p1)

# right
dev.new()
p2 <- ggplot(data.frame(switched, arsenic), aes(arsenic, switched)) +
    geom_jitter(position = position_jitter(width = 0.2, height = 0.01)) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- cbind(1, 0, x, 0*x) %*% beta.mean.1))) +
    stat_function(fun = function(x)
                  1 / (1 + exp(- cbind(1, 0.5, x, 0.5*x) %*% beta.mean.1))) +
    annotate("text", x = c(1.7,2.5), y = c(0.78, 0.56),
             label = c("if dist = 0", "if dist = 50"), size = 4) +
    scale_x_continuous("Arsenic concentration in well water",
                       breaks = seq(from = 0, by = 2, length.out = 5)) +
    scale_y_continuous("Pr(switching)", breaks = seq(0, 1, 0.2))
print(p2)

### Adding social predictors

## With community organization variable
## Model: switched ~ c_dist100 + c_arsenic + c_dist100:c_arsenic + assoc + educ4
## educ4 <- educ / 4

if (!exists("wells_daae_c.sm")) {
    if (file.exists("wells_daae_c.sm.RData")) {
        load("wells_daae_c.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_daae_c.stan", model_name = "wells_daae_c")
        wells_daae_c.sm <- stan_model(stanc_ret = rt)
        save(wells_daae_c.sm, file = "wells_daae_c.sm.RData")
    }
}

data.list.2 <- c("N", "switched", "dist", "arsenic", "assoc", "educ")
wells_daae_c.sf <- sampling(wells_daae_c.sm, data.list.2)
print(wells_daae_c.sf)

## Without community organization variable
## Model: switched ~ c_dist100 + c_arsenic + c_dist100:c_arsenic + educ4

if (!exists("wells_dae_c.sm")) {
    if (file.exists("wells_dae_c.sm.RData")) {
        load("wells_dae_c.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dae_c.stan", model_name = "wells_dae_c")
        wells_dae_c.sm <- stan_model(stanc_ret = rt)
        save(wells_dae_c.sm, file = "wells_dae_c.sm.RData")
    }
}

data.list.3 <- c("N", "switched", "dist", "arsenic", "educ")
wells_dae_c.sf <- sampling(wells_dae_c.sm, data.list.3)
print(wells_dae_c.sf)

## Adding further interactions (centering education variable)
## Model: switched ~ c_dist100 + c_arsenic + c_educ4 + c_dist100:c_arsenic
##                   + c_dist100:c_educ4 + c_arsenic:c_educ4
## c_educ4 <- educ/4 - mean(educ/4)

if (!exists("wells_dae_inter_c.sm")) {
    if (file.exists("wells_dae_inter_c.sm.RData")) {
        load("wells_dae_inter_c.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("wells_dae_inter_c.stan", model_name = "wells_dae_inter_c")
        wells_dae_inter_c.sm <- stan_model(stanc_ret = rt)
        save(wells_dae_inter_c.sm, file = "wells_dae_inter_c.sm.RData")
    }
}

wells_dae_inter_c.sf <- sampling(wells_dae_inter_c.sm, data.list.3)
print(wells_dae_inter_c.sf)
