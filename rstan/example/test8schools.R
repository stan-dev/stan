library(rstan)

model_name <- "_8chools";

m <- stan.model(file = "8schools_bda.stan",
                model.name = model_name, 
                verbose = TRUE)  


dat <- list(J = 8L, 
            y = c(28,  8, -3,  7, -1,  1, 18, 12),
            sigma_y = c(15, 10, 16, 11,  9, 11, 10, 18),
            sigma_xi = 25); 

sampler <- new(m@.modelmod$sampler, dat)

n.iter <- 520

# s1 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools1.csv", seed = 3, chain_id = 1))
# s2 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools2.csv", seed = 3, chain_id = 1))
# s3 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools3.csv", seed = 3, chain_id = 1))
# s4 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools4.csv", seed = 3, chain_id = 1))

# HMC
ss1 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, leapfrog_steps = 5, refresh = -1)  

# NUTS 1 
ss2 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, unit_mass_matrix = TRUE, refresh = -1) 

# NUTS 2 
ss3 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, unit_mass_matrix = TRUE, refresh = -1) 

print(ss) 
plot(ss)
traceplot(ss)
