library(rstan)

### Data

source("mesquite.data.R", echo = TRUE)

### First model: weight ~ diam1 + diam2 + canopy.height + total.height + density + group
data.list <- c("N", "weight", "diam1", "diam2", "canopy_height", "total_height",
               "density", "group")
mesquite.sf <- stan(file='mesquite.stan', data=data.list,
                    iter=1000, chains=4)
print(mesquite.sf)

### Log model: log(weight) ~ log(diam1) + log(diam2) + log(canopy.height)
###                          + log(total.height) + log(density) + group

mesquite_log.sf <- stan(file='mesquite_log.stan', data=data.list,
                        iter=1000, chains=4)
print(mesquite_log.sf)

### Volume model: log(weight) ~ log(canopy_volume)
# canopy_volume <- diam1 * diam2 * canopy_height

mesquite_volume.sf <- stan(file='mesquite_volume.stan', data=data.list,
                           iter=1000, chains=4)
print(mesquite_volume.sf)

### Volume, area & shape model:
# log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape)
#               + log(total.height) + log(density) + group
# canopy_volume <- diam1 * diam2 * canopy_height
# canopy_area   <- diam1 * diam2
# canopy_shape  <- diam1 / diam2

mesquite_vas.sf <- stan(file='mesquite_vas.stan', data=data.list,
                        iter=1000, chains=4)
print(mesquite_vas.sf)

### Last two models

# log(weight) ~ log(canopy_volume) + log(canopy_area) + group

mesquite_va.sf <- stan(file='mesquite_va.stan', data=data.list,
                       iter=1000, chains=4)
print(mesquite_va.sf)

# log(weight) ~ log(canopy_volume) + log(canopy_area) + log(canopy_shape)
#               + log(total_height) + group

mesquite_vash.sf <- stan(file='mesquite_vash.stan', data=data.list,
                         iter=1000, chains=4)
print(mesquite_vash.sf)
