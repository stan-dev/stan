library(rstan)

model_name <- "_8chools";
sfile <- "../../src/models/misc/eight_schools/eight_schools_bda.stan"
m <- stan.model(file = sfile, 
                model.name = model_name, 
                verbose = TRUE)  

dat <- list(J = 8L, 
            y = c(28,  8, -3,  7, -1,  1, 18, 12),
            sigma_y = c(15, 10, 16, 11,  9, 11, 10, 18))

# sampler <- new(m@.modelmod$sampler, dat)
# s1 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools1.csv", seed = 3, chain_id = 1))
# s2 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools2.csv", seed = 3, chain_id = 1))
# s3 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools3.csv", seed = 3, chain_id = 1))
# s4 <- sampler$call_sampler(list(iter = n.iter, thin = 1, sample_file = "8schools4.csv", seed = 3, chain_id = 1))

n.iter <- 5020
# HMC
ss1 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, leapfrog_steps = 5, refresh = 100)  

# NUTS 1 
ss2 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, equal_step_sizes = TRUE, refresh = 100) 

# NUTS 2 
ss3 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, equal_step_sizes = FALSE, refresh = 100) 

print(ss1) 
print(ss2) 
print(ss3) 
plot(ss1)
traceplot(ss1)

ss4 <- sampling(m, data = dat, n.iter = n.iter, n.chains = 4, refresh = 10) 

ss <- stan(sfile, data = dat, n.iter = n.iter, n.chains = 4, sample.file = '8schools.csv')
print(ss)

ss.inits <- ss@inits 
ss.same <- stan(sfile, data = dat, n.iter = n.iter, n.chains = 4, seed = ss@stan.args[[1]]$seed, 
               init.t = 'user', init.v = ss.inits, sample.file = 'ya8schools.csv') 

b <- identical(ss@sim$samples, ss.same@sim$samples) 
# b is not true as ss is initialized randomly while ss.same is not. 


s <- summary(ss.same, pars = "mu_theta", probs = c(.3, .8))
print(ss.same, pars = 'theta', probs = c(.4, .8))
print(ss.same)


fit2 <- stan(sfile, data = dat, n.iter = n.iter, n.chains = 3, pars = "theta")
print(fit2)
print(fit2, pars = 'theta')
print(fit2, pars = 'mu_theta')



