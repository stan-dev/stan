library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq

if (!exists("kidiq_interaction.sm")) {
    if (file.exists("kidiq_interaction.sm.RData")) {
        load("kidiq_interaction.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_interaction.stan", model_name = "kidiq_interaction")
        kidiq_interaction.sm <- stan_model(stanc_ret = rt)
        save(kidiq_interaction.sm, file = "kidiq_interaction.sm.RData")
    }
}

data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_interaction.sf <- sampling(kidiq_interaction.sm, data.list)
print(kidiq_interaction.sf, pars = c("beta", "sigma", "lp__"))

### Centering by subtracting the mean

if (!exists("kidiq_interaction_c.sm")) {
    if (file.exists("kidiq_interaction_c.sm.RData")) {
        load("kidiq_interaction_c.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_interaction_c.stan", model_name = "kidiq_interaction_c")
        kidiq_interaction_c.sm <- stan_model(stanc_ret = rt)
        save(kidiq_interaction_c.sm, file = "kidiq_interaction_c.sm.RData")
    }
}

kidiq_interaction_c.sf <- sampling(kidiq_interaction_c.sm, data.list)
print(kidiq_interaction_c.sf)

### Using a conventional centering point:
# c2_mom_hs <- mom_hs - 0.5
# c2_mom_iq <- mom_iq - 100

if (!exists("kidiq_interaction_c2.sm")) {
    if (file.exists("kidiq_interaction_c2.sm.RData")) {
        load("kidiq_interaction_c2.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_interaction_c2.stan", model_name = "kidiq_interaction_c2")
        kidiq_interaction_c2.sm <- stan_model(stanc_ret = rt)
        save(kidiq_interaction_c2.sm, file = "kidiq_interaction_c2.sm.RData")
    }
}

kidiq_interaction_c2.sf <- sampling(kidiq_interaction_c2.sm, data.list)
print(kidiq_interaction_c2.sf)

### Centering by subtracting the mean & dividing by 2 sd

if (!exists("kidiq_interaction_z.sm")) {
    if (file.exists("kidiq_interaction_z.sm.RData")) {
        load("kidiq_interaction_z.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("kidiq_interaction_z.stan", model_name = "kidiq_interaction_z")
        kidiq_interaction_z.sm <- stan_model(stanc_ret = rt)
        save(kidiq_interaction_z.sm, file = "kidiq_interaction_z.sm.RData")
    }
}

kidiq_interaction_z.sf <- sampling(kidiq_interaction_z.sm, data.list)
print(kidiq_interaction_z.sf)
