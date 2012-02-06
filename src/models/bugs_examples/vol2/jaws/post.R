library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE) 
summary(as.mcmc(post)) 
plot(as.mcmc(post)) 


library(BUGSExamples);
pars <- c("beta0", "beta1", "mu", "Omega"); 
ex <- list(name = "Jaws", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda); 




