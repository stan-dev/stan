library(rstan)

### Data

source("kidiq.data.R", echo = TRUE)

### Model: kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq

data.list <- c("N", "kid_score", "mom_hs", "mom_iq")
kidiq_interaction <- stan(file='kidiq_interaction.stan', data=data.list,
                          iter=1000, chains=4)
print(kidiq_interaction, pars = c("beta", "sigma", "lp__"))

### Centering by subtracting the mean
kidiq_interaction_c <- stan(file='kidiq_interaction_c.stan', data=data.list,
                            iter=1000, chains=4)
print(kidiq_interaction_c)

### Using a conventional centering point:
# c2_mom_hs <- mom_hs - 0.5
# c2_mom_iq <- mom_iq - 100

kidiq_interaction_c2 <- stan(file='kidiq_interaction_c2.stan', data=data.list,
                             iter=1000, chains=4)
print(kidiq_interaction_c2)

### Centering by subtracting the mean & dividing by 2 sd
kidiq_interaction_z <- stan(file='kidiq_interaction_z.stan', data=data.list,
                            iter=1000, chains=4)
print(kidiq_interaction_z)
