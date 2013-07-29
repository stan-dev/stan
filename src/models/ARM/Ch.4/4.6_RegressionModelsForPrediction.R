library(rstan)

### Data

source("mesquite.data.R", echo = TRUE)

### First model: weight ~ diam1 + diam2 + canopy.height + total.height + density + group

if (!exists("mesquite.sm")) {
    if (file.exists("mesquite.sm.RData")) {
        load("mesquite.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite.stan", model_name = "mesquite")
        mesquite.sm <- stan_model(stanc_ret = rt)
        save(mesquite.sm, file = "mesquite.sm.RData")
    }
}

data.list <- c("N", "weight", "diam1", "diam2", "canopy_height", "total_height",
               "density", "group")
mesquite.sf <- sampling(mesquite.sm, data.list)
print(mesquite.sf)

### Log model: log(weight) ~ log(diam1) + log(diam2) + log(canopy.height)
###                          + log(total.height) + log(density) + group

if (!exists("mesquite_log.sm")) {
    if (file.exists("mesquite_log.sm.RData")) {
        load("mesquite_log.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite_log.stan", model_name = "mesquite_log")
        mesquite_log.sm <- stan_model(stanc_ret = rt)
        save(mesquite_log.sm, file = "mesquite_log.sm.RData")
    }
}

mesquite_log.sf <- sampling(mesquite_log.sm, data.list)
print(mesquite_log.sf)

### Volume model: log(weight) ~ log(canopy_volume)
# canopy_volume <- diam1 * diam2 * canopy_height

if (!exists("mesquite_volume.sm")) {
    if (file.exists("mesquite_volume.sm.RData")) {
        load("mesquite_volume.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite_volume.stan", model_name = "mesquite_volume")
        mesquite_volume.sm <- stan_model(stanc_ret = rt)
        save(mesquite_volume.sm, file = "mesquite_volume.sm.RData")
    }
}

mesquite_volume.sf <- sampling(mesquite_volume.sm, data.list)
print(mesquite_volume.sf)

### Volume, area & shape model:
# log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape)
#               + log(total.height) + log(density) + group
# canopy_volume <- diam1 * diam2 * canopy_height
# canopy_area   <- diam1 * diam2
# canopy_shape  <- diam1 / diam2

if (!exists("mesquite_vas.sm")) {
    if (file.exists("mesquite_vas.sm.RData")) {
        load("mesquite_vas.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite_vas.stan", model_name = "mesquite_vas")
        mesquite_vas.sm <- stan_model(stanc_ret = rt)
        save(mesquite_vas.sm, file = "mesquite_vas.sm.RData")
    }
}

mesquite_vas.sf <- sampling(mesquite_vas.sm, data.list)
print(mesquite_vas.sf)

### Last two models

# log(weight) ~ log(canopy_volume) + log(canopy_area) + group

if (!exists("mesquite_va.sm")) {
    if (file.exists("mesquite_va.sm.RData")) {
        load("mesquite_va.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite_va.stan", model_name = "mesquite_va")
        mesquite_va.sm <- stan_model(stanc_ret = rt)
        save(mesquite_va.sm, file = "mesquite_va.sm.RData")
    }
}

mesquite_va.sf <- sampling(mesquite_va.sm, data.list)
print(mesquite_va.sf)

# log(weight) ~ log(canopy_volume) + log(canopy_area) + log(canopy_shape)
#               + log(total_height) + group

if (!exists("mesquite_vash.sm")) {
    if (file.exists("mesquite_vash.sm.RData")) {
        load("mesquite_vash.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("mesquite_vash.stan", model_name = "mesquite_vash")
        mesquite_vash.sm <- stan_model(stanc_ret = rt)
        save(mesquite_vash.sm, file = "mesquite_vash.sm.RData")
    }
}

mesquite_vash.sf <- sampling(mesquite_vash.sm, data.list)
print(mesquite_vash.sf)
