library(rstan)
library(ggplot2)
source("mesquite.data.R")    
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/mesquite

## First model (mesquite.stan)
## lm (weight ~ diam1 + diam2 + canopy.height + total.height + density + group)
if (!file.exists("mesquite.sm.RData")) {
    rt <- stanc("mesquite.stan", model_name="mesquite")
    mesquite.sm <- stan_model(stanc_ret=rt)
    save(mesquite.sm, file="mesquite.sm.RData")
} else {
    load("mesquite.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, LeafWt=LeafWt, Diam1=Diam1,Diam2=Diam2,CanHt=CanHt,TotHt,Dens=Dens,Group=Group)
mesquite.sf1 <- sampling(mesquite.sm, dataList.1)
print(mesquite.sf1)

## Data summary
summary(mesquite)
IQR(Diam1)
IQR(Diam2)
IQR(CanHt)
IQR(TotHt)
IQR(Dens)
IQR(Group)

## Other models

# Log model (mesquite_log.stan)
# lm (log(weight) ~ log(diam1) + log(diam2) + log(canopy.height) + log(total.height) + log(density) + group)
if (!file.exists("mesquite_log.sm.RData")) {
    rt <- stanc("mesquite_log.stan", model_name="mesquite_log")
    mesquite_log.sm <- stan_model(stanc_ret=rt)
    save(mesquite_log.sm, file="mesquite_log.sm.RData")
} else {
    load("mesquite_log.sm.RData", verbose=TRUE)
}

mesquite_log.sf1 <- sampling(mesquite_log.sm, dataList.1)
print(mesquite_log.sf1)

# Volume model (mesquite_canopy_volume.stan)
# lm (log(weight) ~ log(canopy.volume))
if (!file.exists("mesquite_canopy_volume.sm.RData")) {
    rt <- stanc("mesquite_canopy_volume.stan", model_name="mesquite_canopy_volume")
    mesquite_canopy_volume.sm <- stan_model(stanc_ret=rt)
    save(mesquite_canopy_volume.sm, file="mesquite_canopy_volume.sm.RData")
} else {
    load("mesquite_canopy_volume.sm.RData", verbose=TRUE)
}

dataList.2 <- list(N=N, LeafWt=LeafWt, Diam1=Diam1,Diam2=Diam2,CanHt=CanHt)
mesquite_canopy_volume.sf1 <- sampling(mesquite_canopy_volume.sm, dataList.2)
print(mesquite_canopy_volume.sf1)

# Volume, area & shape model (mesquite_volume_area_shape.stan)
# lm (log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape) + log(total.height) + log(density) + group)
if (!file.exists("mesquite_volume_area_shape.sm.RData")) {
    rt <- stanc("mesquite_volume_area_shape.stan", model_name="mesquite_volume_area_shape")
    mesquite_volume_area_shape.sm <- stan_model(stanc_ret=rt)
    save(mesquite_volume_area_shape.sm, file="mesquite_volume_area_shape.sm.RData")
} else {
    load("mesquite_volume_area_shape.sm.RData", verbose=TRUE)
}

mesquite_volume_area_shape.sf1 <- sampling(mesquite_volume_area_shape.sm, dataList.1)
print(mesquite_volume_area_shape.sf1)

# Last two models (mesquite_volume_area.stan)
# lm (log(weight) ~ log(canopy.volume) + log(canopy.area) + group)
if (!file.exists("mesquite_volume_area.sm.RData")) {
    rt <- stanc("mesquite_volume_area.stan", model_name="mesquite_volume_area")
    mesquite_volume_area.sm <- stan_model(stanc_ret=rt)
    save(mesquite_volume_area.sm, file="mesquite_volume_area.sm.RData")
} else {
    load("mesquite_volume_area.sm.RData", verbose=TRUE)
}

dataList.3 <- list(N=N, LeafWt=LeafWt, Diam1=Diam1,Diam2=Diam2,CanHt=CanHt,Group=Group)
mesquite_volume_area.sf1 <- sampling(mesquite_volume_area.sm, dataList.3)
print(mesquite_volume_area.sf1)

# (mesquite_all.stan)
# lm (log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape) + log(total.height) + group)
if (!file.exists("mesquite_all.sm.RData")) {
    rt <- stanc("mesquite_all.stan", model_name="mesquite_all")
    mesquite_all.sm <- stan_model(stanc_ret=rt)
    save(mesquite_all.sm, file="mesquite_all.sm.RData")
} else {
    load("mesquite_all.sm.RData", verbose=TRUE)
}

mesquite_all.sf1 <- sampling(mesquite_all.sm, dataList.1)
print(mesquite_all.sf1)
