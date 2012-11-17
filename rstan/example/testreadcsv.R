
library(rstan)

csvfiles <- c('8schools_1.csv', '8schools_2.csv', 
              '8schools_3.csv', '8schools_4.csv') 

f1 <- rstan:::read_stan_csv(csvfiles)
print(f1)

f2 <- try(stan(fit = f1))

ai <- get_adaptation_info(f1)
cat(ai[[1]])

mode <- get_cppo_mode(f1)
print(mode)

inits <- get_inits(f1)
print(inits)

lp <- get_logposterior(f1)
head(lp[[1]])

sp <- get_sampler_params(f1)

s <- get_seed(f1)
ss <- get_seeds(f1)

a <- get_stancode(f1)
sm <- get_stanmodel(f1)

plot(f1)
traceplot(f1)

# 
model_name <- "_8chools";
sfile <- "../../src/models/misc/eight_schools/eight_schools.stan"

dat <- list(J = 8L, 
            y = c(28,  8, -3,  7, -1,  1, 18, 12),
            sigma = c(15, 10, 16, 11,  9, 11, 10, 18))
iter <- 1002
ss_hmc <- stan(file = sfile, data = dat, iter = iter, chains = 4,
               leapfrog_steps = 5, refresh = -100, 
               sample_file = '8schools_hmc.csv')  
ss_nuts1 <- stan(fit = ss_hmc, data = dat, iter = iter, chains = 4,
                 equal_step_sizes = TRUE, refresh = -1, 
                 sample_file = '8schools_nuts1.csv')

fit_hmc <- read_stan_csv(paste("8schools_hmc_", 1:4, ".csv", sep = ''))
fit_nuts1 <- read_stan_csv(paste("8schools_nuts1_", 1:4, ".csv", sep = ''))

e1 <- extract(fit_hmc)
e2 <- extract(fit_hmc, permuted = FALSE)
dim(e2)
