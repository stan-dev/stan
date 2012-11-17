# 
library(coda)
I <- 4
J <- 2
K <- 5
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')[, -(1:3)]

tmpfun <- function(x, y) {
  paste("[", x, ", ", y, "]", sep = '');
} 

# summary(as.mcmc(post[-(1:1000),]))  ## for save_warmup=1
summary(as.mcmc(post)) 

library(BUGSExamples);
pars <- c("alpha", "beta", "gamma"); 
ex <- list(name = "Alligators", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);


# note that in the stan version of this example, 
# alpha[1] corresponds to alpha[2] in JAGS in which alpha[1] = 0,
# and the same for other parameters. 

library(rstan)

fit <- rstan:::read_stan_csv('samples.csv')
print(fit, digits = 4)


