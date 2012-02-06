library(coda)
post <- read.csv(file = 'samples.csv', header = TRUE) 

poi <- post[, c("mu_beta.1", "mu_beta.2", "sigmasq_y")]  
poi[, 3] <- sqrt(poi[, 3]);
colnames(poi)[3] <- "sigma"; 
summary(as.mcmc(poi)) 

library(BUGSExamples);
pars <- c("mu.beta", "sigma"); 
ex <- list(name = "BiRats", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda); 




