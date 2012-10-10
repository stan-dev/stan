library(rstan)

model_name <- "_8chools3";
sfile <- "../../src/models/misc/eight_schools/eight_schools.stan"

sc1 <- stanc(sfile, obfuscate_model_name = FALSE) 
yam <- stan_model(file = sfile, 
                  model_name = model_name, 
                  save_dso = FALSE, 
                  verbose = TRUE,
                  obfuscate_model_name = FALSE) 
dat <- list(J = 8L, 
            y = c(28,  8, -3,  7, -1,  1, 18, 12),
            sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

iter <- 121

# NUTS 2 
ss3 <- sampling(m, data = dat, iter = iter, chains = 4, equal_step_sizes = FALSE, refresh = 100) 
ainfo3 <- get_adaptation_info(ss3)
lp3 <- get_logposterior(ss3)
yalp3 <- get_logposterior(ss3, inc_warmup = FALSE)
sp3 <- get_sampler_params(ss3)
yasp3 <- get_sampler_params(ss3, inc_warmup = FALSE) 

fit2 <- stan(sfile, data = dat, iter = iter, chains = 3, pars = "theta", obfuscate_model_name = !TRUE)



