
library(coda)
library(BUGSExamples);

pars <- c("q", "beta0C", "beta", "phi") 
ex <- list(name = "Cervix", parameters = pars,
           nSample = 10000, nBurnin = 1000, nThin = 1,
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS')
summary(jagspost$coda)
plot(jagspost$coda);
