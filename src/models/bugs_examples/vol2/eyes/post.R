library(coda) 
post <- read.csv(file = 'samples.csv', header = TRUE, comment.char = '#')[, -(1:3)] 
summary(as.mcmc(post)) 

library(BUGSExamples);
pars <- c("P", "sigma", "lambda"); 
ex <- list(name = "Eyes", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda); 

