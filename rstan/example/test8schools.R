library(rstan)

model_name <- "_8chools";
sfile <- "../../src/models/misc/eight_schools/eight_schools.stan"
m <- stan_model(file = sfile, 
                model_name = model_name, 
                verbose = TRUE)  
m@dso 

yam <- stan_model(file = sfile, 
                  model_name = model_name, 
                  save_dso = FALSE, 
                  verbose = TRUE)  
yam@dso 

dat <- list(J = 8L, 
            y = c(28,  8, -3,  7, -1,  1, 18, 12),
            sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

iter <- 5020
# HMC
ss1 <- sampling(m, data = dat, iter = iter, chains = 4, leapfrog_steps = 5, refresh = 100)  
ainfo1 <- get_adaptation_info(ss1)
lp1 <- get_logposterior(ss1)
yalp1 <- get_logposterior(ss1, inc_warmup = FALSE)
sp1 <- get_sampler_params(ss1) 
yasp1 <- get_sampler_params(ss1, inc_warmup = FALSE) 
gm1 <- get_posterior_mean(ss1)
print(gm1)


# NUTS 1 
ss2 <- sampling(m, data = dat, iter = iter, chains = 4, equal_step_sizes = TRUE, refresh = 100) 
ainfo2 <- get_adaptation_info(ss2)
lp2 <- get_logposterior(ss2)
yalp2 <- get_logposterior(ss2, inc_warmup = FALSE)
sp2 <- get_sampler_params(ss2)
yasp2 <- get_sampler_params(ss2, inc_warmup = FALSE) 
gm2 <- get_posterior_mean(ss2)
print(gm2)

# NUTS 2 
ss3 <- sampling(m, data = dat, iter = iter, chains = 4, equal_step_sizes = FALSE, refresh = 100) 
ainfo3 <- get_adaptation_info(ss3)
lp3 <- get_logposterior(ss3)
yalp3 <- get_logposterior(ss3, inc_warmup = FALSE)
sp3 <- get_sampler_params(ss3)
yasp3 <- get_sampler_params(ss3, inc_warmup = FALSE) 

gm3 <- get_posterior_mean(ss3)
print(gm3)

# Non-diag 
ss4 <- sampling(m, data = dat, iter = iter, chains = 4, nondiag_mass = TRUE, refresh = 100) 
ainfo4 <- get_adaptation_info(ss4)
lp4 <- get_logposterior(ss4)
yalp4 <- get_logposterior(ss4, inc_warmup = FALSE)
sp4 <- get_sampler_params(ss4)
yasp4 <- get_sampler_params(ss4, inc_warmup = FALSE) 

gm4 <- get_posterior_mean(ss4)
print(gm4)

print(ss1) 
print(ss2) 
print(ss3) 
plot(ss1)
traceplot(ss1)

ss9 <- sampling(m, data = dat, iter = iter, chains = 4, refresh = 10) 

iter <- 52012
ss <- stan(sfile, data = dat, iter = iter, chains = 4, sample_file = '8schools.csv')
print(ss)

ss.inits <- ss@inits 
ss.same <- stan(sfile, data = dat, iter = iter, chains = 4, seed = ss@stan_args[[1]]$seed, 
                init = ss.inits, sample_file = 'ya8schools.csv') 

b <- identical(ss@sim$samples, ss.same@sim$samples) 
# b is not true as ss is initialized randomly while ss.same is not. 


s <- summary(ss.same, pars = "mu", probs = c(.3, .8))
print(ss.same, pars = 'theta', probs = c(.4, .8))
print(ss.same)


# fit2 <- stan(sfile, data = dat, iter = iter, chains = 3, pars = "theta")
# print(fit2)
# print(fit2, pars = 'theta')
# print(fit2, pars = 'mu')



