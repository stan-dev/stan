library(coda) 
# library(BUGSExamples)
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#') 

pars <- c("a.1", "a.2", "a.3", "beta", "pi", "kappa", "sigma"); 
poi <- post[, pars]; 
summary(as.mcmc(poi)) 


### It seems that JAGS does not support the version as specified as in BUGS
### regarding to the restriction on a's. 
#   ex <- list(name = "Inhalers", parameters = c("a", "beta", "kappa", "tau", "sigma", "log.sigma"), 
#              nSample = 2000, nBurnin = 500, nThin = 1, 
#              nChain = 3)
#    
#   jagspost <- runExample(ex, engine = 'JAGS') 
#   summary(jagspost$coda) 
